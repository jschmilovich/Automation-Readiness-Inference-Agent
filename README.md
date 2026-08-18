# Automation Readiness Inference Agent

A retrieval-augmented agent that reads a workflow description and scores how ready it is to automate. It scores the workflow across eight dimensions, estimates a value and complexity rating, maps it to an autonomy tier (T0 to T4), and returns a confidence level and the similar past workflows it used to calibrate. It runs as a FastAPI web app with single-workflow and batch scoring.

## Setup (Windows, no admin required)

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Mac or Linux, activate the venv with `source .venv/bin/activate` and use `pip install -r requirements.txt`.

The first run downloads the embedding and reranker models once (needs internet that one time), then they run offline on CPU.

## Model access

Scoring calls a Claude model through an internal LLM gateway. `score.py` mints a short-lived key at call time. Model access must be configured for scoring to run. Retrieval and the smoke tests run locally with no key.

## Run the web app

```bash
.venv\Scripts\python.exe -m uvicorn app:app --port 8000
```

Then open http://127.0.0.1:8000 in your browser on the same machine. This is a local address, it serves the app on whatever computer is running it, so each person runs their own copy. Single mode scores one workflow with its eight dimensions, evidence, tier, and neighbors. Batch mode scores many at once and ranks them into ready to automate, needs refinement, and keep human.

## Run the evaluation

```bash
.venv\Scripts\python.exe evaluate.py
```

Scores every labeled workflow, compares against the reference labels, and reports exact accuracy, adjacent agreement, binary F1, Cohen's kappa, and a credible interval, for complexity and value.

## Files

- `workflows.csv`  the labeled workflow set, used as both the retrieval pool and the benchmark
- `corpus.py`  loads the workflow set into clean records
- `retrieval.py`  hybrid retrieval, dense embeddings + BM25 + cross-encoder reranker fused with Reciprocal Rank Fusion
- `score.py`  builds the prompt, calls the model, returns structured scores across the eight dimensions
- `evaluate.py`  the evaluation harness and its statistics
- `eval_retrieval.py`  measures retrieval quality against a label-concordance proxy
- `sc_experiment.py`  self-consistency experiment, scores each workflow multiple times and compares
- `app.py`  the FastAPI web app
- `static/index.html`  the web app front end
- `run_smoke.py`, `run_hybrid.py`  quick local checks that retrieval works
