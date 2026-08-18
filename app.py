# Automation Readiness Inference Agent -- local web app (backend)
"""FastAPI wrapper around the existing scoring engine.

Reuses the real engine unchanged:
  corpus.load_corpus / corpus.Workflow      -- the workflow record + loader
  retrieval.HybridRetriever(.retrieve)      -- neighbor retrieval
  score.score_workflow(..., use_stub=False) -- scoring through the model gateway
  score.RUBRIC / score._saxecap_label       -- display helpers

The 39 existing workflows are the RETRIEVAL POOL only. Users score NEW pasted
workflows against that pool; the pool entries are never selectable or re-scored.

Version 1: single-workflow scoring at POST /api/score.
Version 2: batch submit + ranked/bucketed results at POST /api/score_batch
(streams NDJSON progress). Both share the core score_description() below.
"""
import re
import json
from copy import deepcopy
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from corpus import load_corpus, Workflow
from retrieval import HybridRetriever
from score import score_workflow, RUBRIC, _saxecap_label

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Neighbors to calibrate against -- matches the k used in evaluate.py.
NEIGHBORS_K = 5

# Engine handles, populated once at startup (see lifespan).
ENGINE = {"corpus": None, "retriever": None}

# In-memory result cache: exact workflow input -> scored result dict. Keyed by the
# (name, department, description) triple so the same input always returns the same
# result instead of re-hitting the model. Serves both single and batch scoring
# (both go through score_description). Process-local; cleared on restart. Does not
# touch scoring logic, the engine, or workflows.csv.
_RESULT_CACHE: dict[tuple[str, str, str], dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the corpus and build the retriever ONCE. Building the retriever loads
    # the embedding + reranker models, so we never do it per request.
    corpus = load_corpus()
    ENGINE["corpus"] = corpus
    ENGINE["retriever"] = HybridRetriever(corpus)
    print(f"[startup] retrieval pool ready: {len(corpus)} workflows")
    yield
    ENGINE.clear()


app = FastAPI(title="Automation Readiness Inference Agent", lifespan=lifespan)


class ScoreRequest(BaseModel):
    description: str
    name: str = ""
    department: str = ""


def _neighbor_view(neighbors) -> list[dict]:
    """Shape retrieved (Workflow, relevance) pairs for the page: name,
    department, and their SaxeCap complexity/value labels. Neighbors stay in
    retrieval order (most similar first); the raw reranker score is intentionally
    not exposed -- it is meaningless (and sometimes negative) to a user."""
    return [
        {
            "name": w.name or "(unnamed)",
            "department": w.department or "—",
            "complexity": _saxecap_label(w.complexity),
            "value": _saxecap_label(w.utility),
        }
        for w, _rel in neighbors
    ]


def _dimension_view(score) -> list[dict]:
    """All eight rubric dimensions in rubric order, each with its 1..5 score and
    the evidence line the model returned."""
    evidence = score.evidence or {}
    return [
        {
            "key": key,
            "definition": definition,
            "score": score.dimensions.get(key),
            "evidence": evidence.get(key, ""),
        }
        for key, definition in RUBRIC.items()
    ]


def score_description(description: str, name: str = "", department: str = "") -> dict:
    """Core scoring path -- reused by V1 (single) and, later, V2 (batch).

    Builds a NEW Workflow from user text, retrieves neighbors from the existing
    pool, and scores through the real model gateway (use_stub=False).
    """
    text = (description or "").strip()
    if not text:
        raise ValueError("Please paste a workflow description to score.")

    # Cache check: same exact input (name + department + description) -> stored
    # result, so we never re-score identical text. deepcopy so callers can't mutate
    # the cached object. Key on the raw args so single and batch share entries.
    cache_key = (name, department, description)
    cached = _RESULT_CACHE.get(cache_key)
    if cached is not None:
        return deepcopy(cached)

    retriever = ENGINE["retriever"]
    if retriever is None:
        raise RuntimeError("Engine is still starting up; try again in a moment.")

    # wid=-1 so it never collides with the pool ids (0..N-1); labels blank
    # because they are exactly what we are predicting.
    new_wf = Workflow(
        wid=-1,
        department=department.strip(),
        name=name.strip(),
        description=text,
        utility="",
        complexity="",
        roi="",
        source="user",
    )

    neighbors = retriever.retrieve(new_wf.text, k=NEIGHBORS_K, exclude_wid=None)
    result = score_workflow(new_wf, neighbors, use_stub=False)

    view = {
        "readiness": result.readiness,
        "tier": result.tier,
        "confidence": result.confidence,
        "complexity": result.complexity,
        "value": result.utility,
        "dimensions": _dimension_view(result),
        "neighbors": _neighbor_view(neighbors),
    }
    # Store a copy so a later mutation of the returned dict can't corrupt the cache.
    _RESULT_CACHE[cache_key] = deepcopy(view)
    return view


@app.post("/api/score")
def api_score(req: ScoreRequest):
    """Score one pasted workflow. On any failure, return the error text (HTTP
    200, ok=false) so the page can show it instead of crashing."""
    try:
        return {"ok": True, "result": score_description(req.description, req.name, req.department)}
    except Exception as e:  # gateway/parse/etc. -- surface it to the page
        return JSONResponse(status_code=200, content={"ok": False, "error": str(e)})


# --- Version 2: batch scoring, ranked + bucketed ----------------------------
# Reuses score_description() (same engine, same retrieval, same real gateway).
# Streams NDJSON so the page shows real per-workflow progress, since scoring
# several takes time. V1 is untouched.

# Bucket thresholds. readiness/confidence are 0..1; dimension scores are 1..5.
READINESS_HIGH = 0.66      # at/above -> ready to automate
READINESS_LOW = 0.33       # below    -> keep human
CONFIDENCE_LOW = 0.5       # below    -> needs refinement
HUMAN_JUDGMENT_HIGH = 4    # at/above -> keep human (a person is needed)

BUCKET_ORDER = ("ripe_now", "needs_data_work", "keep_human")


class BatchRequest(BaseModel):
    text: str


def parse_batch(text: str) -> list[dict]:
    """Split pasted text into workflow blocks separated by blank lines. Each
    block may start with optional 'Name:' and/or 'Department:' lines (either
    order); the remaining lines are the description."""
    blocks = re.split(r"\n\s*\n", (text or "").strip())
    items = []
    for block in blocks:
        lines = block.splitlines()
        name, dept, i = "", "", 0
        while i < len(lines):
            m_name = re.match(r"\s*name\s*:\s*(.+)", lines[i], re.I)
            m_dept = re.match(r"\s*dep(?:artmen)?t\s*:\s*(.+)", lines[i], re.I)
            if m_name and not name:
                name = m_name.group(1).strip(); i += 1; continue
            if m_dept and not dept:
                dept = m_dept.group(1).strip(); i += 1; continue
            break
        description = "\n".join(lines[i:]).strip()
        if not (description or name):
            continue
        items.append({"name": name, "department": dept, "description": description})
    return items


def _bucket(readiness: float, confidence: float, human_judgment) -> str:
    """Assign a bucket. 'Keep human' is a safety override (low readiness OR a
    person's judgment is needed); then 'needs refinement' (mid readiness OR the
    model isn't confident); otherwise 'ready to automate'."""
    hj_high = human_judgment is not None and human_judgment >= HUMAN_JUDGMENT_HIGH
    if readiness < READINESS_LOW or hj_high:
        return "keep_human"
    if readiness < READINESS_HIGH or confidence < CONFIDENCE_LOW:
        return "needs_data_work"
    return "ripe_now"


def _top_reasons(dimensions: list[dict], n: int = 2) -> list[dict]:
    """The n highest-scoring dimensions with their evidence lines -- the headline
    reasons shown per result."""
    ranked = sorted(dimensions, key=lambda d: (d.get("score") or 0), reverse=True)
    return [
        {"dimension": d["key"], "score": d.get("score"), "reason": d.get("evidence") or ""}
        for d in ranked[:n]
    ]


def _batch_stream(text: str):
    """Score each parsed workflow, emitting NDJSON progress lines, then a final
    grouped + ranked result line."""
    items = parse_batch(text)
    total = len(items)
    yield json.dumps({"type": "start", "total": total}) + "\n"

    scored = []
    for idx, it in enumerate(items, 1):
        name = it["name"] or f"Workflow {idx}"
        try:
            r = score_description(it["description"], it["name"], it["department"])
            hj = next((d.get("score") for d in r["dimensions"]
                       if d["key"] == "human_judgment"), None)
            scored.append({
                "name": name,
                "department": it["department"],
                "readiness": r["readiness"],
                "tier": r["tier"],
                "value": r["value"],
                "complexity": r["complexity"],
                "confidence": r["confidence"],
                "human_judgment": hj,
                "top_reasons": _top_reasons(r["dimensions"]),
                "bucket": _bucket(r["readiness"], r["confidence"], hj),
                "error": None,
            })
        except Exception as e:  # one failure must not kill the batch
            scored.append({"name": name, "department": it["department"], "error": str(e)})
        yield json.dumps({"type": "progress", "done": idx, "total": total, "name": name}) + "\n"

    ok = [e for e in scored if not e.get("error")]
    errors = [e for e in scored if e.get("error")]
    ok.sort(key=lambda e: e["readiness"], reverse=True)   # rank by readiness
    groups = {b: [] for b in BUCKET_ORDER}
    for e in ok:
        groups[e["bucket"]].append(e)
    yield json.dumps({"type": "result", "groups": groups, "errors": errors,
                      "total": total, "scored": len(ok)}) + "\n"


@app.post("/api/score_batch")
def api_score_batch(req: BatchRequest):
    """Version 2: score many workflows, stream progress, return ranked buckets."""
    return StreamingResponse(_batch_stream(req.text), media_type="application/x-ndjson")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
