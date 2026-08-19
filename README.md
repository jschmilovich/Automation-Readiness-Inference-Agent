# Automation Readiness Inference Agent

A retrieval-augmented agent that reads a workflow description and scores how ready it is to automate. It scores across eight dimensions, estimates a value and complexity rating, maps the workflow to an autonomy tier (T0 to T4), and returns a confidence level and the similar past workflows it used to calibrate. It runs as a FastAPI web app with single-workflow and batch scoring.

## IMPORTANT: model access is required to score

Scoring calls a Claude model through Asurion's internal LLM gateway. `score.py` mints a short-lived key at call time using a local helper. This means:

- The app must be run on a machine with Asurion gateway access configured (an Asurion work machine). It will not score on a personal machine, because the key helper and gateway are internal to Asurion.
- Retrieval, the web page, and the local test scripts run anywhere with no key. Only the scoring step needs the gateway.
- If scoring returns an error like "Unexpected end of JSON input" or a missing key-helper path, that means gateway access is not set up on that machine. That is the cause, not a bug in the app.

Set up gateway access first, then everything below works.

## Setting up gateway access (one-time)

1. Go to the internal gateway page and follow its install instructions: **https://llmgateway.asurion53.com/**. Installing Claude Code from here is the simplest path, it comes with the gateway model access built in.

2. Once installed, it authenticates you to the gateway and sets up the local key helper that mints a short-lived key for each model call. `score.py` finds this helper automatically on the current machine.

3. The gateway allows a set of Claude models. This app uses `claude-opus-4-8`. If your access does not include it, `score.py` can be pointed at another allowed model.

If you already use Claude Code through the gateway on your machine, scoring should work with no extra setup. If scoring fails with a missing key-helper path, gateway access is not yet configured on that machine, start at the link above.

## Setup

On an Asurion Windows machine:

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Mac or Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The first run downloads the embedding and reranker models once (needs internet that one time), then they run offline on CPU.

## Run the web app

Windows:

```bash
.venv\Scripts\python.exe -m uvicorn app:app --port 8000
```

Mac or Linux:

```bash
.venv/bin/python -m uvicorn app:app --port 8000
```

Then open **http://127.0.0.1:8000** in a browser on the same machine. This is a local address, it serves the app on whatever computer is running it, so each person runs their own copy.

Do NOT open `static/index.html` directly or through the VS Code Live Server preview. That loads the page with no backend behind it, and every score will fail with a JSON error. The app only works when started with the command above.

Single mode scores one workflow and shows the eight dimensions with evidence, the retrieved neighbors and their labels, the tier, and the confidence. Batch mode scores many workflows at once and ranks them into ready to automate, needs refinement, and keep human.

## Run the evaluation

Windows:

```bash
.venv\Scripts\python.exe evaluate.py
```

Mac or Linux:

```bash
.venv/bin/python evaluate.py
```

Scores every labeled workflow, compares against the reference labels, and reports exact accuracy, adjacent agreement, binary F1, Cohen's kappa, and a credible interval, for complexity and value.

## Files

- `workflows.csv`  the labeled workflow set, used as both the retrieval pool and the benchmark
- `corpus.py`  loads the workflow set into clean records
- `retrieval.py`  hybrid retrieval, dense embeddings + BM25 + cross-encoder reranker fused with Reciprocal Rank Fusion
- `score.py`  builds the prompt, calls the model through the gateway, returns structured scores across the eight dimensions
- `evaluate.py`  the evaluation harness and its statistics
- `eval_retrieval.py`  measures retrieval quality against a label-concordance proxy
- `sc_experiment.py`  self-consistency experiment, scores each workflow multiple times and compares
- `app.py`  the FastAPI web app
- `static/index.html`  the web app front end
- `run_smoke.py`, `run_hybrid.py`  quick local checks that retrieval works
