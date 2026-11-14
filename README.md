# Agent harness

A small OpenAI-powered loop: **propose numeric parameters → evaluate locally →
keep improvements → repeat**. Tasks supply the objective, initial parameters,
and evaluator. The harness has no built-in dataset or model architecture.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
python agent.py --task examples/target/task.json --rounds 5
```

The included example is a tiny arithmetic task demonstrating the interface.
Try the local setup without an API key or any installed dependencies:

```bash
python3 agent.py --task examples/target/task.json --dry-run
python3 -m unittest test_agent.py
```

## Task contract

A task JSON supplies:

```json
{
  "objective": "Explain what the parameters should achieve.",
  "parameters": {"weight": 0.0, "bias": 0.0},
  "context": "Feature definitions, constraints, or other relevant evidence.",
  "evaluator": ["python3", "evaluate.py"]
}
```

The evaluator runs relative to the task file's directory. It receives the
parameter object as JSON on stdin and must emit one JSON object on stdout:

```json
{"score": 0.42, "feedback": {"errors": "Details useful for the next proposal"}}
```

**Higher scores are better.** Send diagnostics to stderr. Evaluators are
user-provided programs; generated agent output is never executed as code.
Choose a deterministic evaluation setup and keep dataset splits inside your
own evaluator. The harness does not implement sampling or training.

Each API request includes the objective, context, current best parameters,
evaluator results, and the last three attempts. Its structured response is:

```json
{
  "rationale": "Why this change might help.",
  "edits": [{"parameter": "weight", "value": 0.5}]
}
```

Edits set absolute values; omitted parameters stay unchanged. Only known
parameters and finite numeric values are accepted. Improvements are selected
by score. This is adaptive model selection, not an unbiased validation score.

## Options and artifacts

- `--model`: defaults to `OPENAI_MODEL`, then `gpt-6-astra`.
- `--rounds`: 5 API requests by default, with no automatic retries.
- `--max-output-tokens`: 6,000 per request by default; not a dollar budget.
- `--eval-timeout`: evaluator timeout in seconds, default 60.
- `--output`: new output directory; defaults to `runs/<timestamp>`.
- `--dry-run`: evaluate baseline and save the prompt/schema, without API calls.
- `--evaluate-final`: run the task's optional `final_evaluator` command once,
  after freezing the winner. Its results are never fed back to the agent.

Run directories contain the task/config, proposal inputs, per-round results,
API token usage, `best_parameters.json`, `best_metrics.json`, and a frozen
parameter hash. The task file is never overwritten. API or evaluator failures
stop the run, leaving the saved best parameters intact. Invalid proposals are
rejected and fed back. Existing output directories are refused.

The API key is read from the environment, never included in prompts or logs,
and removed from the evaluator's environment. `.env` files are not loaded
automatically. API calls use the official Responses API with structured JSON
output. Live API access requires your own key; tests use mocked responses.
