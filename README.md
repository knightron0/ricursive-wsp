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
| `tools(code)` | the last 10,000 characters of stdout/stderr of Python run in the sandbox | sandbox |
| `end_tools()` | `reasoned` only: ends phase 1 for good | harness |
| `predict(weights, indices, layer, channels)` | one layer's output, per image, of the network a diff defines: `fc` logits with the predicted class, or any module's output averaged over space (one value per channel), optionally a `[start, end)` channel range; ≤500 images and ≤6,000 numbers per call, 3 significant digits, never labels | harness |
| `evaluate(weights)` | accuracy, loss, per-class accuracy and confusion of a diff on the training subset (aggregates only); in `reasoned`, the accuracy alone | harness |
| `submit(weights)` | validation result; a valid diff ends the episode | harness |

`predict()`, `evaluate()` and `submit()` check the diff's format first (tensor names, number counts, finite values, the
`--max-numbers` budget) and list every problem; a call that fails the check is logged with status `format` and
nothing is scored or submitted.

## Turns

An episode has `--max-turns` turns (default 10); a turn is one model reply, which may hold several calls that run in
order. The prompt states the limit, and the last result of every turn says how many turns are left (and, before the
last turn, that it must contain a valid `submit()`). A reply whose calls were all format rejections does not count as
a turn, for at most `--format-retries` replies (default 3). No valid `submit()` by the end of the last turn means no
score.

## Modes

| Mode | Calls | What sandbox code can read |
|---|---|---|
| `blind` | `tools`, `submit` | no data |
| `reasoned` | phase 1: `tools`, `data` and `predict` (pool); **`end_tools()` permanently disables `tools()`**; phase 2: `labels`, `evaluate`, `data` and `predict` (any image), `submit` | pool images only |
| `full` | `data`, `labels`, `tools`, `predict`, `evaluate`, `submit` | training images and labels |

No gradients in any mode: autograd, `torch.optim` and `torch.func` transforms are disabled in the sandbox.

`reasoned` splits the training subset into two disjoint, class-balanced parts (`--labeled-share`, default 0.5):
- **pool** (indices `0..P-1`): unlabeled images. The sandbox holds these, and `data()` shows them. Their labels are never returned by any call.
- **labeled set** (indices `P..N-1`): `labels()` covers only these, and `evaluate()` scores on them (accuracy only). Their pixels (`data()`) and representations (`predict()`) become visible only in phase 2.

The episode runs in two phases. Phase 1 has `tools()`, and `data()` and `predict()` on the pool; `labels()`,
`evaluate()` and `submit()` are refused. The agent calls `end_tools()` to enter phase 2, which is one-way: `tools()`
is gone for good, and `labels()`, `evaluate()`, `data()`, `predict()` and `submit()` become available. So nothing
derived from a label ever reaches code: every number the agent's code computes describes a pool image, and every
label, labeled image, labeled-image representation and score is seen only in context. `predict()` runs the agent's
own network (a diff it writes) in the harness, not agent code. This closes two exploits seen in earlier runs:
precomputing a fit in code and finishing it once the labels arrive, and probing `evaluate()` with crafted diffs,
pasting the confusion matrices into `tools()` and solving for class statistics.

Phase 2 is meant for in-context learning from labels. The system prompt and the `end_tools()` result say so: call
`predict()` on labeled images to get the network's representations (final logits or any intermediate layer),
`labels()` on the same indices, and reason over the (representation, label) pairs to decide the numbers to write.
In `reasoned`, `evaluate()` returns only the accuracy: no confusion matrix, per-class accuracy or loss.

What's checked mechanically, by the harness and by `audit.json` after each run:
1. In `reasoned`, no `tools()` call executes after `end_tools()`, and no `labels()`, `evaluate()` or `submit()` executes before it.
2. In `reasoned`, `labels()` never returns a pool label, and no labeled-set image or representation (`data()`, `predict()`) is shown before `end_tools()`.
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
| `wsp/episode.py` | the calls, the mode rules, the `reasoned` phases, the audit |
| `wsp/agent.py` | Anthropic agent (streaming, adaptive thinking, prompt caching, server-side refusal fallback), scripted agent, episode loop |
| `wsp/prompt.py` | system and user prompt, per-mode rules, tool definitions |
| `wsp/sandbox.py` | runs one `tools()` call |
| `wsp/tasks.py` | the two tasks, start weights, diff validation |
| `wsp/data.py`, `model.py`, `evaluation.py`, `toolkit.py` | splits (45k train / 5k val / test), ResNet-18, scoring, sandbox helpers |

Each run writes `runs/<timestamp>/`: `config.json`, `subset_indices.json`, `prompt.json`,
`calls.json` (every call with its phase and status: `ok`, `refused` by the mode or phase rules, or `failed` on
invalid input; results longer than 2,000 characters keep their head and tail), `transcript.json`, `usage.json`,
`audit.json`, `workspace/`, and once there's a valid submission, `weights.pt` and `result.json`. `calls.json`,
`transcript.json` and `usage.json` are rewritten after every turn; results the model hasn't been sent yet appear as a
final `pending_tool_results` (or `pending_user_message`) entry of `transcript.json`. Stdout gets one line per turn: that turn's calls
with their status, the phase (`reasoned`), wall time, and tokens and cost when the provider reports them, e.g.
`turn 36 [phase 1->2]: tools, end_tools | 81s | in 52.0k (50.0k cached), out 3.1k, $0.061`.
A diff with several invalid tensors is rejected with all of them listed.
