# Automation Readiness Inference Agent -- self-consistency experiment (report-only)
"""One-off experiment: compare single-call scoring to self-consistency scoring.

Report-only. Changes NOTHING in the pipeline: it imports the existing metric
functions (report, extra_report, norm, ORDINAL) from evaluate.py and the existing
score_workflow from score.py, and only adds a majority-vote wrapper on top.

For each workflow we call the model 5 times with the existing prompt.
  - single-call prediction  = the 1st of the 5 calls (same conditions baseline)
  - self-consistency pred    = majority vote across all 5 labels, per dimension,
                               ties broken by the median on the Low<Medium<High scale
Then both prediction sets are run through the SAME report()/extra_report().

~195 model calls total (39 workflows x 5), roughly 25 minutes.
"""
from collections import Counter

from corpus import load_corpus
from retrieval import HybridRetriever
from score import score_workflow
from evaluate import report, extra_report, norm, ORDINAL

N = 5
INV = {0: "Low", 1: "Medium", 2: "High"}


def majority_vote(labels):
    """Majority vote over Low/Medium/High labels; ties broken by the median on
    the ordinal Low<Medium<High scale."""
    labs = [norm(l) or l for l in labels]
    counts = Counter(labs)
    top = max(counts.values())
    winners = [lab for lab, c in counts.items() if c == top]
    if len(winners) == 1:
        return winners[0]
    ords = sorted(ORDINAL.get(l, 1) for l in labs)  # median tie-break
    return INV[ords[len(ords) // 2]]


if __name__ == "__main__":
    wfs = load_corpus()
    print(f"loaded {len(wfs)} workflows; self-consistency N={N} ({N}x model calls)\n")
    print("building retriever, scoring every workflow 5x (real model)...\n")
    retr = HybridRetriever(wfs)

    single_cx, single_ut = {}, {}
    sc_cx, sc_ut = {}, {}
    for w in wfs:
        neighbors = retr.retrieve(w.text, k=5, exclude_wid=w.wid)
        cx_votes, ut_votes = [], []
        for _ in range(N):
            s = score_workflow(w, neighbors, use_stub=False)
            cx_votes.append(s.complexity)
            ut_votes.append(s.utility)
        single_cx[w.wid], single_ut[w.wid] = cx_votes[0], ut_votes[0]
        sc_cx[w.wid] = majority_vote(cx_votes)
        sc_ut[w.wid] = majority_vote(ut_votes)
        print(f"  scored wid={w.wid:>3}  cx {cx_votes} -> {sc_cx[w.wid]:6s}"
              f"  ut {ut_votes} -> {sc_ut[w.wid]}", flush=True)

    print("\n################ SINGLE-CALL (1st of the 5) ################\n")
    report("complexity", single_cx, wfs)
    extra_report("complexity", single_cx, wfs)
    report("utility", single_ut, wfs)
    extra_report("utility", single_ut, wfs)

    print("\n################ SELF-CONSISTENCY (majority of 5) ################\n")
    report("complexity", sc_cx, wfs)
    extra_report("complexity", sc_cx, wfs)
    report("utility", sc_ut, wfs)
    extra_report("utility", sc_ut, wfs)
