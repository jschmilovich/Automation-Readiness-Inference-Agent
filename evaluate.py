# Automation Readiness Inference Agent
"""Evaluation harness: score every workflow, compare to the SaxeCap labels,
report the headline number.

Runs today on STUB scores (no key), so the whole pipeline works end to end. When
you wire the model call in score.py, these same numbers become real. Nothing else
changes.

Reports, for complexity and value:
  accuracy            how often the agent matches SaxeCap
  95% credible range  an honest band around that number, since the set is small
  Cohen's kappa       agreement adjusted for chance
Plus scoring latency (p50 / p95).
"""
import time
import numpy as np
from scipy.stats import beta
from sklearn.metrics import cohen_kappa_score, f1_score
from corpus import load_corpus
from retrieval import HybridRetriever
from score import score_workflow
from eval_retrieval import concordance, chance_level


def norm(label: str) -> str:
    l = label.strip().lower()
    if l in ("med", "medium"): return "Medium"
    if l == "low": return "Low"
    if l in ("high", "very high"): return "High"
    return ""  # blank, skipped


def credible_interval(correct: int, n: int, mass: float = 0.95):
    """Beta-Binomial credible interval around the accuracy. Posterior is
    Beta(correct+1, wrong+1); we read off the central `mass` band."""
    tail = (1 - mass) / 2
    a, b = correct + 1, (n - correct) + 1
    return beta.ppf(tail, a, b), beta.ppf(1 - tail, a, b)


def report(field: str, preds: dict, workflows) -> None:
    true, pred = [], []
    for w in workflows:
        gold = norm(getattr(w, field))
        if not gold:
            continue
        true.append(gold)
        pred.append(preds[w.wid])
    n = len(true)
    correct = sum(t == p for t, p in zip(true, pred))
    acc = correct / n
    lo, hi = credible_interval(correct, n)
    kappa = cohen_kappa_score(true, pred)
    print(f"=== {field} vs SaxeCap  (n={n}) ===")
    print(f"  accuracy          : {acc:.0%}   ({correct}/{n})")
    print(f"  95% credible range: {lo:.0%} to {hi:.0%}")
    print(f"  Cohen's kappa     : {kappa:.2f}   (agreement adjusted for chance)")
    print()


# --- report-only extras (add new numbers; change no scoring, no existing metric) ---
ORDINAL = {"Low": 0, "Medium": 1, "High": 2}


def _pairs(field, preds, workflows):
    """Same true/pred filtering report() uses, factored out for the extra metrics."""
    true, pred = [], []
    for w in workflows:
        gold = norm(getattr(w, field))
        if not gold:
            continue
        true.append(gold)
        pred.append(preds[w.wid])
    return true, pred


def extra_report(field, preds, workflows):
    """Adjacent agreement + binary worth-automating accuracy/F1. Report-only."""
    true, pred = _pairs(field, preds, workflows)
    n = len(true)
    # (1) adjacent: correct if within one bucket on the Low < Medium < High scale
    adj = sum(abs(ORDINAL[t] - ORDINAL.get(norm(p), 1)) <= 1
              for t, p in zip(true, pred)) / n
    # (2) binary worth-automating: High/Medium -> 1, Low -> 0
    tb = [0 if t == "Low" else 1 for t in true]
    pb = [0 if norm(p) == "Low" else 1 for p in pred]
    bacc = sum(a == b for a, b in zip(tb, pb)) / n
    bf1 = f1_score(tb, pb, pos_label=1, zero_division=0)
    print(f"  [{field}] adjacent agreement : {adj:.0%}   (within one bucket)")
    print(f"  [{field}] binary accuracy    : {bacc:.0%}   (worth-automating H/M vs L)")
    print(f"  [{field}] binary F1          : {bf1:.2f}   (worth-automating = positive)")
    print()


if __name__ == "__main__":
    wfs = load_corpus()
    print(f"loaded {len(wfs)} workflows\n")
    print("building retriever, scoring every workflow (STUB scores until the key is wired)...\n")
    retr = HybridRetriever(wfs)

    preds_cx, preds_ut, latencies = {}, {}, []
    for w in wfs:
        neighbors = retr.retrieve(w.text, k=5, exclude_wid=w.wid)
        t0 = time.perf_counter()
        s = score_workflow(w, neighbors, use_stub=True)
        latencies.append((time.perf_counter() - t0) * 1000)
        preds_cx[w.wid] = s.complexity
        preds_ut[w.wid] = s.utility

    print("*** STUB scores, not real. Wire the model call in score.py for the real number. ***\n")
    report("complexity", preds_cx, wfs)
    extra_report("complexity", preds_cx, wfs)
    report("utility", preds_ut, wfs)
    extra_report("utility", preds_ut, wfs)

    print("--- retrieval recall @k (label-concordance proxy, report-only) ---")
    for field in ("complexity", "utility"):
        r3 = concordance(retr, wfs, field, k=3)
        r5 = concordance(retr, wfs, field, k=5)
        print(f"  {field:10s}: recall@3 {r3:.0%}   recall@5 {r5:.0%}"
              f"   (chance {chance_level(wfs, field):.0%})")
    print()

    lat = np.array(latencies)
    print(f"scoring latency   : p50 {np.percentile(lat, 50):.2f} ms   p95 {np.percentile(lat, 95):.2f} ms")
    print("(near zero now, since scoring is stubbed; real latency shows once the model call is live)")
