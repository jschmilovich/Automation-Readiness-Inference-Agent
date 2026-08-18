# Automation Readiness Inference Agent
"""Retrieval quality number. No API key, no new labels.

Idea: a good retriever pulls neighbors that RESEMBLE the query, and every
workflow already carries SaxeCap labels (complexity, value). So we measure how
often the retrieved neighbors share the query's label, versus how often that
would happen by chance.

Metric: label concordance @ k
  For each workflow used as the query, look at its top-k neighbors and compute
  the fraction that carry the SAME label. Average over all queries. Then compare
  to the chance level (how often two workflows would match at random).
  Above chance means retrieval is pulling workflows that actually resemble the
  query, which is what a scoring agent needs from it.
"""
from collections import Counter
from corpus import load_corpus
from retrieval import DenseRetriever, HybridRetriever


def norm(label: str) -> str:
    l = label.strip().lower()
    if l in ("med", "medium"): return "medium"
    if l == "low": return "low"
    if l in ("high", "very high"): return "high"
    return ""  # blank / unknown, skipped


def concordance(retriever, workflows, field="complexity", k=3) -> float:
    labels = {w.wid: norm(getattr(w, field)) for w in workflows}
    queries = [w for w in workflows if labels[w.wid]]        # only labeled queries
    per_query = []
    for w in queries:
        hits = retriever.retrieve(w.text, k=k, exclude_wid=w.wid)
        neigh = [h for h, _ in hits if labels[h.wid]]        # neighbors that have a label
        if not neigh:
            continue
        same = sum(1 for h in neigh if labels[h.wid] == labels[w.wid])
        per_query.append(same / len(neigh))
    return sum(per_query) / len(per_query) if per_query else 0.0


def chance_level(workflows, field="complexity") -> float:
    labels = [norm(getattr(w, field)) for w in workflows]
    labels = [l for l in labels if l]
    n = len(labels)
    if n < 2:
        return 0.0
    counts = Counter(labels)
    return sum((counts[l] - 1) / (n - 1) for l in labels) / n


if __name__ == "__main__":
    wfs = load_corpus()
    print(f"loaded {len(wfs)} workflows\n")
    print("building retrievers (uses the models you already downloaded)...\n")
    dense = DenseRetriever(wfs)
    hybrid = HybridRetriever(wfs)

    for field in ("complexity", "utility"):
        labeled = sum(1 for w in wfs if norm(getattr(w, field)))
        print(f"=== {field}  ({labeled} of {len(wfs)} workflows labeled) ===")
        print(f"  chance level : {chance_level(wfs, field):.0%}   (random match rate)")
        print(f"  dense  @3    : {concordance(dense, wfs, field, k=3):.0%}")
        print(f"  hybrid @3    : {concordance(hybrid, wfs, field, k=3):.0%}")
        print()
    print("Above chance = retrieval pulls workflows that resemble the query.")
