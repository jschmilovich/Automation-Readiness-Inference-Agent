# Automation Readiness Inference Agent
"""Compare dense-only vs hybrid retrieval on one workflow, so you can see the
reranker change the order. No API key needed.

Note: the hybrid scores are cross-encoder relevance scores, on a different scale
than the dense cosine scores, so compare the ORDER of results, not the numbers.
"""
from corpus import load_corpus
from retrieval import DenseRetriever, HybridRetriever

wfs = load_corpus()
print(f"loaded {len(wfs)} workflows\n")

query_wf = wfs[0]
print(f"QUERY: [{query_wf.wid}] {query_wf.name}  ({query_wf.department})\n")
print("building retrievers (downloads the reranker model once)...\n")

dense = DenseRetriever(wfs)
hybrid = HybridRetriever(wfs)

print("DENSE only (meaning match):")
for w, s in dense.retrieve(query_wf.text, k=3, exclude_wid=query_wf.wid):
    print(f"  {s:.3f}  [{w.wid}] {w.name}  ({w.department})")

print("\nHYBRID (dense + BM25 + reranker):")
for w, s in hybrid.retrieve(query_wf.text, k=3, exclude_wid=query_wf.wid):
    print(f"  {s:.3f}  [{w.wid}] {w.name}  ({w.department})")
