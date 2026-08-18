# Automation Readiness Inference Agent
"""Smoke test: prove retrieval works on the real data, no API key needed.

Picks one workflow, retrieves its nearest neighbors, and prints them so you
can eyeball whether the matches are sensible.
"""
from corpus import load_corpus
from retrieval import DenseRetriever

wfs = load_corpus()
print(f"loaded {len(wfs)} workflows\n")

retr = DenseRetriever(wfs)

query_wf = wfs[0]
print(f"QUERY: [{query_wf.wid}] {query_wf.name}  ({query_wf.department})\n")
print("nearest workflows:")
for w, score in retr.retrieve(query_wf.text, k=3, exclude_wid=query_wf.wid):
    print(f"  {score:.3f}  [{w.wid}] {w.name}  ({w.department})")
