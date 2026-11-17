# ricursive-wsp: weight-space programming

An agent writes the weights of a ResNet-18 as **literal numbers**. Its one answer is a diff over
the initialized network: the tensors, or blocks of tensors, it wants to change, with every value
or a constant fill. Everything it leaves out keeps its start values. The submitted network is scored once on held-out val (and test), which
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

A diff is a list of entries applied in order (a later entry overwrites an earlier one where they overlap):

| Entry | Sets |
|---|---|
| `{"name", "values"}` | the whole tensor, every number, flat in row-major order |
| `{"name", "index", "values"}` | a block: `index` is one `[start, stop)` pair per leading axis (later axes whole); `values` are the block's numbers, row-major |
| `{"name", "fill": c}`, `{"name", "index", "fill": c}` | every element of the tensor or block to `c` |

Toward the `--max-numbers` budget, every number in `values` counts and each fill counts as 1. Null keys count as
absent (the OpenAI backend sends strict schemas, so unused keys arrive as null).

`predict()`, `evaluate()` and `submit()` check the diff's format first (tensor names, block ranges, number counts,
finite values, the `--max-numbers` budget) and list every problem, entry by entry; a call that fails the check is logged with status `format` and
nothing is scored or submitted.

## Turns

An episode has `--max-turns` turns (default 10); a turn is one model reply, which may hold several calls that run in
order. The prompt states the limit, and the last result of every turn says how many turns are left (and, before the
last turn, that it must contain a valid `submit()`). A reply whose calls were all format rejections does not count as
a turn, for at most `--format-retries` replies (default 3). No valid `submit()` by the end of the last turn means no
score.

With `--provider claude-sdk` (the episode runs inside a Claude Agent SDK loop) there is no turn limit unless
`--max-turns` is given, and the prompt states a soft output-token budget (`--token-budget`). With `--max-turns N`
every model response counts as a turn, including one with no calls, one whose calls were all rejected, and one cut
off at the per-reply output limit. The prompt states the limit, every call result ends with a
`[harness] turn k of N` note, and the harness stops the loop when reply N+1 begins. `--sdk-max-turns` is only a
backstop on Claude Code's own turn count (default 3 × `--max-turns`). `--usage-limit F` holds the next call while the
five-hour plan window is at least F used, until it resets (tool calls may then wait for hours; the timeout is raised
to 6 h), so the agent pauses without losing a turn; holds go to `usage_holds.json`.

## Modes

| Mode | Calls | What sandbox code can read |
|---|---|---|
| `blind` | `tools`, `submit` | no data |
| `reasoned` | phase 1: `tools`, `data`, `predict`; **`end_tools()` permanently disables `tools()`**; phase 2: `labels`, `evaluate`, `data`, `predict`, `submit` | every subset image, no labels |
| `full` | `data`, `labels`, `tools`, `predict`, `evaluate`, `submit` | training images and labels |

No gradients in any mode: autograd, `torch.optim` and `torch.func` transforms are disabled in the sandbox.

`reasoned` uses one training subset, with no split. Phase 1 has no labels: the sandbox holds every subset image
(`data.pt` = images and class names), and `tools()`, `data()` and `predict()` work on any of them; `labels()`,
`evaluate()` and `submit()` are refused. The agent calls `end_tools()` to enter phase 2, which is one-way: `tools()`
is gone for good, and `labels()` (any subset image), `evaluate()` (accuracy on the whole subset), `data()`,
`predict()` and `submit()` are available. So every image can have its pixels, its representations and its label
seen, but nothing derived from a label ever reaches code: labels appear only after code is gone, and are combined
with images and representations only in the agent's own context. `predict()` runs the agent's own network (a diff it
writes) in the harness, not agent code. Phase-1 code can prepare features for exactly the images whose labels the
agent later reads; finishing a fit from those labels can only happen in context. Probing `evaluate()` with crafted
diffs and solving for class statistics in code is impossible, since `evaluate()` exists only in phase 2.

Phase 2 is meant for in-context learning from labels. The system prompt and the `end_tools()` result say so: call
`predict()` on images to get the network's representations (final logits or any intermediate layer), `labels()` on
the same indices, and reason over the (representation, label) pairs to decide the numbers to write. In `reasoned`,
`evaluate()` returns only the accuracy: no confusion matrix, per-class accuracy or loss.

What's checked mechanically, by the harness and by `audit.json` after each run:
1. In `reasoned`, no `tools()` call executes after `end_tools()`, and no `labels()`, `evaluate()` or `submit()` executes before it.
2. No call outside the mode's list ever executes.
3. The sandbox's `data.pt` holds labels only in `full` (in `reasoned`, images only).
4. The transcript holds no `image` or `document` blocks.

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
between calls. `torch`, `math`, `data` and `start` (the start state_dict) are preloaded, and
nothing else: no helper library, so the agent writes its own forward passes, statistics and
fits. There's no network, no subprocesses, threads or ctypes, and no API keys. Code can read
only the Python install and the workspace (not `wsp/`, so no harness code can be imported or
read), and can write only the workspace. Gradients are
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
| `wsp/data.py`, `model.py`, `evaluation.py` | splits (45k train / 5k val / test), ResNet-18, scoring |

Each run writes `runs/<timestamp>/`: `config.json`, `subset_indices.json`, `prompt.json`,
`calls.json` (every call with its phase and status: `ok`, `refused` by the mode or phase rules, or `failed` on
invalid input; results longer than 2,000 characters keep their head and tail), `transcript.json`, `usage.json`,
`audit.json`, `workspace/`, and once there's a valid submission, `weights.pt` and `result.json`. `calls.json`,
`transcript.json` and `usage.json` are rewritten after every turn; results the model hasn't been sent yet appear as a
final `pending_tool_results` (or `pending_user_message`) entry of `transcript.json`. Stdout gets one line per turn: that turn's calls
with their status, the phase (`reasoned`), wall time, and tokens and cost when the provider reports them, e.g.
`turn 36 [phase 1->2]: tools, end_tools | 81s | in 52.0k (50.0k cached), out 3.1k, $0.061`.
A diff with several invalid entries is rejected with all of them listed, each named by tensor and block (e.g.
`fc.weight [0:10, 37:38]: got 9 numbers, block [10, 1] needs 10`).
