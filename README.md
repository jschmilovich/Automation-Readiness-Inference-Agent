# Automation Readiness Inference Agent

Scores workflows for how ready they are to automate. Reads a workflow, retrieves
similar already-scored ones to ground the call, then (next step) scores it across
the rubric and recommends an autonomy tier.

## Setup (WSL2 / Linux, inside VS Code)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python corpus.py      # confirms the workflows load
python run_smoke.py   # embeds them and prints nearest neighbors for one workflow
```

The first run downloads the embedding model once (needs internet that one time),
then it runs offline on CPU.

## Files
- `workflows.csv`  the scored workflow set (SaxeCap labels = your benchmark answer key)
- `corpus.py`      loads the CSV into clean workflow records
- `retrieval.py`   embeds the workflows and returns nearest neighbors (dense, local)
- `run_smoke.py`   proves retrieval works on the real data

## What's next (needs model access: API key or Claude Code)
- `score.py`     send a workflow + its retrieved neighbors + the rubric to Claude, get structured scores back
- `evaluate.py`  run scoring across all rows, compare to the SaxeCap labels, report agreement (your headline number)
- upgrade `retrieval.py` to hybrid: add BM25 keyword search + a cross-encoder reranker