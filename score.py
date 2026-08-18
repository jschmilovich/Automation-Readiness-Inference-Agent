# Automation Readiness Inference Agent
"""Scoring step: read a workflow plus its retrieved neighbors, return structured scores.

The model call is STUBBED for now, so this runs with no API key. Everything
around it is real: the rubric, the prompt it will send, the parsing, the types.
When you have model access, wire ONE function (call_model) and real scores flow
through with no other change.
"""
import os
import json
import time
import hashlib
import subprocess
from dataclasses import dataclass, field

# The eight dimensions the agent scores, each with a short definition.
RUBRIC = {
    "volume":            "how often the workflow runs",
    "repetitiveness":    "set rules vs case-by-case judgment",
    "input_quality":     "clean structured inputs vs messy",
    "human_judgment":    "how much a person's decision is needed (higher means less automatable)",
    "error_cost":        "how costly or regulated a mistake is",
    "data_availability": "is the source data reachable and current",
    "systems_handoffs":  "how many tools and people it crosses",
    "time_cost":         "how much effort it takes today",
}

TIERS = ["T0", "T1", "T2", "T3", "T4"]   # T0 = human only ... T4 = fully autonomous
BUCKETS = ["Low", "Medium", "High"]


@dataclass
class Score:
    wid: int
    dimensions: dict          # dimension name -> 1..5
    complexity: str           # predicted build complexity: Low / Medium / High
    utility: str              # predicted value: Low / Medium / High
    readiness: float          # overall readiness 0..1
    tier: str                 # T0..T4
    confidence: float         # 0..1
    evidence: dict = field(default_factory=dict)


def _saxecap_label(raw: str) -> str:
    """Normalize a raw SaxeCap complexity/value label for display in the prompt."""
    l = (raw or "").strip().lower()
    if l in ("med", "medium"):
        return "Medium"
    if l == "low":
        return "Low"
    if l in ("high", "very high"):
        return "High"
    return "unlabeled"


def build_prompt(workflow, neighbors) -> str:
    """Assemble the scoring prompt, calibrated to SaxeCap's labeling conventions."""
    dims = "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
    nb = "\n".join(
        f"- {w.name} ({w.department}): "
        f"SaxeCap complexity={_saxecap_label(w.complexity)}, "
        f"value={_saxecap_label(w.utility)}"
        for w, _ in neighbors
    ) or "(none)"
    return f"""You score how ready a business workflow is for automation. You must
calibrate your complexity and value buckets to SaxeCap's conventions so they match
how SaxeCap's own analysts would label this workflow.

Rubric, score each 1 to 5:
{dims}

SaxeCap bucket definitions, match these exactly:
- complexity = estimated engineering BUILD TIME, not how hard it is to run. Estimate
  it the way SaxeCap does: first estimate how many months of engineering effort it
  would take to build this automation, then map the months to a bucket:
    Low    = under 4 months to build
    Medium = 4 to 6 months to build
    High   = over 6 months to build
  Do NOT estimate in isolation. Anchor to the labeled neighbors below: start from the
  build months implied by your nearest neighbors' actual complexity buckets, weight
  the closest neighbors most, and adjust only for concrete differences you can name.
- value = the annual EBITDA UPLIFT from automating the workflow (cost saved plus
  revenue gained per year, in dollars). Estimate the dollar uplift, then map it:
    Low    = under $2M per year
    Medium = $2M to $5M per year
    High   = over $5M per year
  (SaxeCap's "Very High" collapses into High.) Do NOT estimate in isolation. Anchor
  to the labeled neighbors below: start from the dollar range implied by your nearest
  neighbors' actual value buckets, weight the closest neighbors most, and adjust only
  for concrete differences you can name.

Worked examples for complexity, note how the month estimate maps across the bucket
boundaries:
- "Auto-tag incoming support emails using existing labels and a simple classifier
  over clean ticket data." Build estimate ~3 months -> under 4 -> complexity = Low.
- "Reconcile invoices across three disconnected finance systems, building a new data
  pipeline plus a human-in-the-loop review step." Build estimate ~7 months -> over 6
  -> complexity = High.
  (A build of 4 to 6 months would be Medium.)

Worked examples for value, note how the EBITDA-uplift estimate maps across the
boundaries:
- "Automate a niche monthly report used by one small team." Annual EBITDA uplift
  ~$0.8M -> under $2M -> value = Low.
- "Automate fraud triage across the entire claims pipeline, cutting losses and manual
  review at scale." Annual EBITDA uplift ~$8M -> over $5M -> value = High.
  (An uplift of $2M to $5M would be Medium.)

Autonomy tier definitions. Assign the single tier that best fits how this workflow
should run once automated:
- T0 = fully manual: a human does everything.
- T1 = assisted: the tool suggests, but a human does the work.
- T2 = human-in-the-loop: the system does the work and a human reviews and approves
  it each time.
- T3 = human-on-the-loop: the system runs autonomously and a human only spot-checks
  or handles exceptions.
- T4 = fully autonomous: no human review needed.
Choose the tier by matching the workflow against these definitions.

Similar workflows already labeled by SaxeCap. Use their complexity and value labels
as calibration anchors: if the workflow to score resembles a neighbor, its buckets
should land near that neighbor's.
{nb}

Workflow to score:
{workflow.text}

First identify your two nearest anchor neighbors from the list above, then reason
briefly BEFORE committing to each score. Return ONLY a JSON object with these keys,
in this order:
  "nearest_anchors"        list of your 2 closest neighbors, each an object
                           {{"name": ..., "complexity": ..., "value": ...}} copied
                           from their SaxeCap labels above
  "dimensions"             each rubric key -> integer 1..5
  "complexity_reasoning"   one short sentence: estimate the build in months, anchored
                           to the nearest neighbors' build-month buckets
  "estimated_build_months" your numeric build-time estimate, in months
  "complexity"             "Low" / "Medium" / "High", derived from the months above
  "utility_reasoning"      one short sentence: estimate the annual EBITDA uplift in
                           $M, anchored to the nearest neighbors' value buckets
  "estimated_annual_ebitda_uplift_musd"  your numeric estimate, in millions of USD
  "utility"                "Low" / "Medium" / "High", derived from the $ above
  "readiness"              overall readiness 0..1
  "tier"                   one of "T0".."T4", chosen against the tier definitions above
  "confidence"             0..1
  "evidence"               rubric key -> short quote from the workflow"""


# --- Asurion LLM gateway config -------------------------------------------
GATEWAY_BASE_URL = "https://llmgateway.asurion53.com"
GATEWAY_MODEL = "claude-opus-4-8"
API_KEY_HELPER = (
    r"C:\Users\jessica.schmilovich\AppData\Local"
    r"\asurion-llm-gateway\api-key-helper.exe"
)


# Gateway keys last ~1h (CLAUDE_CODE_API_KEY_HELPER_TTL_MS). Cache and re-mint a
# bit early rather than on every call; also force a re-mint if a cached key is
# ever rejected mid-run.
_KEY_TTL_S = 3000
_key_cache = {"key": None, "minted_at": 0.0}


def _mint_key() -> str:
    """Mint a fresh short-lived gateway key via the api-key-helper (stdout)."""
    return subprocess.check_output([API_KEY_HELPER], text=True).strip()


def _get_key(force: bool = False) -> str:
    """Return a cached gateway key, re-minting only when near expiry (or forced)."""
    now = time.time()
    if force or _key_cache["key"] is None or (now - _key_cache["minted_at"]) >= _KEY_TTL_S:
        _key_cache["key"] = _mint_key()
        _key_cache["minted_at"] = now
    return _key_cache["key"]


def _extract_json(text: str) -> str:
    """Pull the JSON object from a reply that may wrap it in fences or prose."""
    t = text.strip()
    if t.startswith("```"):
        t = t[t.find("\n") + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
        t = t.strip()
    # Fall back to the outermost braces if the model adds a preamble/trailer.
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        t = t[start:end + 1]
    return t.strip()


def call_model(prompt: str) -> str:
    """Send `prompt` to the Asurion LLM gateway and return the JSON text reply.

    Uses a cached key (see _get_key) so batch runs don't invoke the helper on
    every call. Retries a few times: re-mints on auth failure, and asks again if
    a reply comes back empty/truncated/unparseable (transient gateway hiccups).
    The import is lazy so the stub path still runs without the anthropic package.
    """
    from anthropic import Anthropic, AuthenticationError

    last = ""
    for attempt in range(3):
        client = Anthropic(base_url=GATEWAY_BASE_URL, api_key=_get_key())
        try:
            resp = client.messages.create(
                model=GATEWAY_MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
        except AuthenticationError:
            _key_cache["key"] = None  # cached key rejected: force a re-mint, retry
            continue
        last = _extract_json("".join(b.text for b in resp.content if b.type == "text"))
        try:
            json.loads(last)
            return last               # valid JSON, done
        except json.JSONDecodeError:
            continue                  # empty/truncated reply: ask again
    return last                       # exhausted retries: let parse_score raise on it


def _stub_response(workflow) -> str:
    """Deterministic placeholder so the pipeline runs with no key.
    NOT a real score. Derived from the workflow id so runs are reproducible."""
    h = int(hashlib.md5(str(workflow.wid).encode()).hexdigest(), 16)
    pick = lambda seq, salt: seq[(h >> salt) % len(seq)]
    dims = {k: 1 + ((h >> (i * 2)) % 5) for i, k in enumerate(RUBRIC)}
    return json.dumps({
        "dimensions": dims,
        "complexity": pick(BUCKETS, 3),
        "utility": pick(BUCKETS, 7),
        "readiness": round(((h >> 11) % 100) / 100, 2),
        "tier": pick(TIERS, 5),
        "confidence": round(0.5 + ((h >> 13) % 50) / 100, 2),
        "evidence": {k: "(stub)" for k in RUBRIC},
    })


def parse_score(wid: int, raw: str) -> Score:
    d = json.loads(raw)
    return Score(
        wid=wid,
        dimensions=d["dimensions"],
        complexity=d["complexity"],
        utility=d["utility"],
        readiness=float(d["readiness"]),
        tier=d["tier"],
        confidence=float(d["confidence"]),
        evidence=d.get("evidence", {}),
    )


def score_workflow(workflow, neighbors, use_stub: bool = True) -> Score:
    # Env override so callers that hardcode use_stub=True (e.g. evaluate.py) can be
    # flipped to real scoring without edits: set ARA_REAL_SCORES=1.
    if os.environ.get("ARA_REAL_SCORES") == "1":
        use_stub = False
    prompt = build_prompt(workflow, neighbors)
    # === MODEL CALL SEAM ===  swap the stub for call_model(prompt) when the key is ready
    raw = _stub_response(workflow) if use_stub else call_model(prompt)
    return parse_score(workflow.wid, raw)
