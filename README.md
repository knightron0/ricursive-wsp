# ricursive-wsp: weight-space programming

An agent writes the weights of a ResNet-18 as **literal numbers**. Its one answer is a diff over
the initialized network: the tensors it wants to change, with every value. Tensors it leaves out
keep their start values. The submitted network is scored once on held-out val (and test), which
the agent never sees. Everything the agent sees is text: no image blocks, no vision encoder.

## Calls

| Call | Returns to the agent | Runs in |
|---|---|---|
| `data(indices)` | training images as hex pixel text (96 rows of 64 hex digits each; ≤16 per call), never labels | harness |
| `labels(indices)` | class labels | harness |
| `tools(code)` | stdout/stderr of Python run in the sandbox | sandbox |
| `evaluate(weights)` | accuracy, loss, per-class accuracy and confusion of a diff on the training subset (aggregates only) | harness |
| `submit(weights)` | validation result; a valid diff ends the episode | harness |

## Modes

| Mode | Calls | What sandbox code can read |
|---|---|---|
| `blind` | `tools`, `submit` | no data |
| `reasoned` | all five, but **the first `labels()` or `evaluate()` permanently disables `tools()`** | pool images only |
| `full` | all five | training images and labels |

No gradients in any mode: autograd, `torch.optim` and `torch.func` transforms are disabled in the sandbox.

`reasoned` splits the training subset into two disjoint, class-balanced parts (`--labeled-share`, default 0.5):
- **pool** (indices `0..P-1`): unlabeled images. The sandbox holds these, and `data()` shows them. Their labels are never returned by any call.
- **labeled set** (indices `P..N-1`): `labels()` covers only these, and `evaluate()` scores on them. Their pixels (`data()`) become visible only after the lock.

`labels()` and `evaluate()` both carry label information, so whichever executes first is the lock: `tools()` is
gone for good, while `labels()`, `evaluate()` and `data()` stay available. So nothing derived from a label ever
reaches code: every number code computes describes a pool image, and every label, labeled image and score is
seen only in context. This closes two exploits seen in earlier runs: precomputing a fit in code and finishing it
once the labels arrive, and probing `evaluate()` with crafted diffs, pasting the confusion matrices into
`tools()` and solving for class statistics.

What's checked mechanically, by the harness and by `audit.json` after each run:
1. In `reasoned`, no `tools()` call executes after the first executed `labels()` or `evaluate()`.
2. In `reasoned`, `labels()` never returns a pool label, and no labeled-set image is shown before the lock.
3. No call outside the mode's list ever executes.
4. The sandbox's `data.pt` holds labels only in `full` (in `reasoned`, pool images only).
5. The transcript holds no `image` or `document` blocks.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python train_base.py            # only for cifar10-to-cifar100: trains the CIFAR-10 source model

export ANTHROPIC_API_KEY=...    # Claude Sonnet 5.5 by default (--model, --effort)
# or --provider openai (OPENAI_API_KEY), or --provider openrouter (OPENROUTER_API_KEY, Opus 5.5 by default)
python -m wsp --task scratch-cifar10 --mode blind
python -m wsp --task scratch-cifar10 --mode reasoned --data-fraction 0.1
python -m wsp --task scratch-cifar10 --mode full     --data-fraction 0.1 --evaluate-test

python -m wsp ... --dry-run                 # write the prompt and tool definitions, no API call
python -m wsp ... --script calls.json       # replay scripted calls, no API call
python -m unittest test_wsp.py              # offline tests
```

A script is a JSON list of turns, each a list of `{"name": ..., "args": {...}}`.

## Sandbox (`wsp/sandbox.py`)

Each `tools()` call is a fresh Python process in the run's `workspace/`, whose files persist
between calls. `torch`, `math`, `toolkit`, `data` and `start` (the start state_dict) are
preloaded. There's no network, no subprocesses, threads or ctypes, and no API keys. Code can read
only the Python install, `wsp/` and the workspace, and can write only the workspace. Gradients are
disabled. The isolation uses Python audit hooks: it's process isolation, **not a security sandbox**.

## Files

| Path | Role |
|---|---|
| `wsp/__main__.py` | CLI: subset, prompt, episode, audit, val/test scoring |
| `wsp/episode.py` | the five calls, the mode rules, the `reasoned` lock, the audit |
| `wsp/agent.py` | Anthropic agent (streaming, adaptive thinking, prompt caching, server-side refusal fallback), scripted agent, episode loop |
| `wsp/prompt.py` | system and user prompt, per-mode rules, tool definitions |
| `wsp/sandbox.py` | runs one `tools()` call |
| `wsp/tasks.py` | the two tasks, start weights, diff validation |
| `wsp/data.py`, `model.py`, `evaluation.py`, `toolkit.py` | splits (45k train / 5k val / test), ResNet-18, scoring, sandbox helpers |

Each run writes `runs/<timestamp>/`: `config.json`, `subset_indices.json`, `prompt.json`,
`calls.json` (every call with its phase, executed or refused), `transcript.json`, `usage.json`,
`audit.json`, `workspace/`, and once there's a valid submission, `weights.pt` and `result.json`.
