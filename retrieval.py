# Automation Readiness Inference Agent
"""Retrieval over the workflow set.

DenseRetriever   - meaning-based only (the simple baseline).
HybridRetriever  - the strong version: dense + keyword, fused, then reranked.

All local, no API key. The hybrid stack has three stages:
  1. Dense retrieval   embeddings, matches on meaning
  2. Keyword retrieval  BM25, matches on exact words and terms
  3. Fuse + rerank      combine both rankings with Reciprocal Rank Fusion,
                        then a cross-encoder reranker reorders the finalists

Dense alone misses exact terms, keyword alone misses meaning, and the reranker
sharpens the final order. This is the standard strong-retrieval setup.
"""
import re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from corpus import Workflow


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class DenseRetriever:
    """Meaning-based retrieval only. Kept as the baseline to compare against."""
    def __init__(self, workflows: list[Workflow], model_name: str = "all-MiniLM-L6-v2"):
        self.workflows = workflows
        self.model = SentenceTransformer(model_name)
        vecs = self.model.encode([w.text for w in workflows], normalize_embeddings=True)
        self.vecs = np.asarray(vecs, dtype=np.float32)

    def retrieve(self, query: str, k: int = 3, exclude_wid: int | None = None):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.vecs @ q
        order = np.argsort(-sims)
        hits = []
        for idx in order:
            w = self.workflows[idx]
            if exclude_wid is not None and w.wid == exclude_wid:
                continue
            hits.append((w, float(sims[idx])))
            if len(hits) == k:
                break
        return hits


def _enrich_text(w: Workflow) -> str:
    """Retrieval text built only from fields available at inference time.

    Leads with department and initiative name (strong signals of what a workflow
    is and roughly how complex/valuable it is), then the description. Deliberately
    EXCLUDES the SaxeCap complexity/value/roi labels: those are the prediction
    target, so indexing them would leak the answer into retrieval and inflate the
    label-concordance proxy. Department is repeated once to upweight it in both the
    embedding and BM25 signals."""
    return f"Department: {w.department}. {w.department}. {w.name}. {w.description}"


class HybridRetriever:
    """Dense + BM25 fused with weighted Reciprocal Rank Fusion, then cross-encoder
    reranked. Text enrichment, fusion weights, reranker model, and candidate-pool
    size are configurable; the defaults are the best combination found on the
    label-concordance proxy (see the sweep in the project history)."""
    def __init__(self, workflows: list[Workflow],
                 dense_model: str = "all-MiniLM-L6-v2",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
                 enrich: bool = True,
                 dense_weight: float = 1.0,
                 bm25_weight: float = 0.5,
                 candidate_k: int = 20,
                 rrf_k: int = 60):
        self.workflows = workflows
        self.enrich = enrich
        self.texts = [(_enrich_text(w) if enrich else w.text) for w in workflows]

        # 1. dense index
        self.embedder = SentenceTransformer(dense_model)
        vecs = self.embedder.encode(self.texts, normalize_embeddings=True)
        self.vecs = np.asarray(vecs, dtype=np.float32)

        # 2. keyword index
        self.bm25 = BM25Okapi([_tokenize(t) for t in self.texts])

        # 3. reranker (downloads once, then offline, runs on CPU)
        self.reranker = CrossEncoder(reranker_model)

        # fusion + candidate-pool config (tunable)
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    @staticmethod
    def _rank_order(scores) -> list[int]:
        return list(np.argsort(-np.asarray(scores)))

    def retrieve(self, query: str, k: int = 3, exclude_wid: int | None = None,
                 candidate_k: int | None = None, rrf_k: int | None = None):
        candidate_k = self.candidate_k if candidate_k is None else candidate_k
        rrf_k = self.rrf_k if rrf_k is None else rrf_k

        # dense ranking
        q_vec = self.embedder.encode([query], normalize_embeddings=True)[0]
        dense_rank = self._rank_order(self.vecs @ q_vec)

        # keyword ranking
        bm25_rank = self._rank_order(self.bm25.get_scores(_tokenize(query)))

        # weighted Reciprocal Rank Fusion: a doc ranked high by either method rises,
        # with dense vs BM25 contributions scaled by their weights
        rrf: dict[int, float] = {}
        for rank_list, weight in ((dense_rank, self.dense_weight),
                                  (bm25_rank, self.bm25_weight)):
            for pos, idx in enumerate(rank_list):
                rrf[idx] = rrf.get(idx, 0.0) + weight * (1.0 / (rrf_k + pos))
        fused = sorted(rrf, key=lambda i: -rrf[i])

        # keep the top candidates, dropping the query workflow itself
        candidates = [i for i in fused
                      if not (exclude_wid is not None and self.workflows[i].wid == exclude_wid)]
        candidates = candidates[:candidate_k]
        if not candidates:
            return []

        # cross-encoder rerank: scores the query against each candidate directly
        pairs = [[query, self.texts[i]] for i in candidates]
        rerank_scores = self.reranker.predict(pairs)
        reranked = sorted(zip(candidates, rerank_scores), key=lambda x: -x[1])

        return [(self.workflows[i], float(s)) for i, s in reranked[:k]]
