# ricursive-wsp: weight-space programming

An LLM **writes the weights of a ResNet-18 directly**, as Python programs, with no
gradient descent, no internet and only X% of the training data.

| Task | Start weights | Data the program gets | Scored on |
|---|---|---|---|
| `scratch-cifar10` | random ResNet-18 init (10 classes) | X% of CIFAR-10 train | CIFAR-10 val / test |
| `cifar10-to-cifar100` | ResNet-18 trained on CIFAR-10, fc reinitialized for 100 classes | X% of CIFAR-100 train | CIFAR-100 val / test |

```
start weights ─► val diagnostics ─► LLM writes build(weights, data, info)
      ▲                                          │
      │                 restricted subprocess: no grads, no network, only the X% subset
      │                                          │
      └── keep if val accuracy improves ◄── evaluate on val
                         ...rounds...
      freeze best.pt (sha256) ─► optionally score start vs best on test, once
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_base.py     # downloads CIFAR-10, trains the CIFAR-10 source model (cifar10-to-cifar100 only)
```

## Run

```bash
export OPENAI_API_KEY=...
python agent.py --task scratch-cifar10 --data-fraction 0.1 --rounds 20 --evaluate-test
python agent.py --task cifar10-to-cifar100 --data-fraction 0.1 --rounds 20 --evaluate-test

# Any OpenRouter model (OpenAI-compatible Chat Completions, strict JSON schema)
export OPENROUTER_API_KEY=...
python agent.py --provider openrouter --model anthropic/claude-opus-5-5 \
    --task scratch-cifar10 --data-fraction 0.1 --rounds 20 --evaluate-test

# No API: preview the first prompt, or run hand-written programs
python agent.py --task scratch-cifar10 --data-fraction 0.1 --dry-run
python agent.py --task scratch-cifar10 --data-fraction 0.1 --programs examples/programs/random_features.py
python agent.py --task cifar10-to-cifar100 --data-fraction 0.1 --programs examples/programs/linear_probe.py
python -m unittest test_agent.py
```

## The program the LLM writes

```python
def build(weights, data, info):
    # weights: current best ResNet-18 state_dict {name: tensor}
    # data:    {"images": uint8 [N,3,32,32], "labels": [N], "classes": [...]}, the X% subset only
    # info:    task, num_classes, data budget, current val metrics
    ...
    return weights   # same names and shapes, finite, running_var >= 0
```

`torch`, `math` and `toolkit` (`wsp/toolkit.py`) are available. The toolkit runs the
network and captures any module's outputs, or the per-channel mean/var needed for
BatchNorm; it also extracts image patches and does ZCA whitening, k-means and
ridge regression.

## Rules the harness enforces

- **Data budget:** `--data-fraction X` picks a fixed, class-balanced subset of the
  45k train split (`--data-seed`). Its indices are saved in the run directory.
  Only that subset is handed to the program.
- **No gradients:** grad mode is off and can't be turned back on; `torch.optim` and
  `torch.func` gradient transforms are removed (`--allow-gradients` lifts this).
- **No internet, no other data:** an audit hook blocks sockets, DNS, subprocesses
  and threads. Programs can read only the Python/torch install, `wsp/` and their
  own temp directory, and can write only there. The API key is removed from their
  environment. The only network traffic is the harness calling the LLM.
  This is best-effort isolation, **not a security sandbox**: generated code runs
  as your user.
- **Splits:** 5k of the official train set is val (the search signal); test is
  loaded only after `best.pt` is frozen, and never shown to the model.
- **Acceptance:** strictly higher val accuracy, ties broken by lower loss. Rounds
  build on the current best. Failed programs (errors, timeouts, bad shapes, NaNs)
  are fed back to the model and the run continues. API failures stop the run.

## Each round the LLM sees

Task description, ResNet-18 layout and state_dict naming, toolkit docs, data
budget, per-tensor stats, val accuracy and loss, per-class accuracy (the worst
and best classes for CIFAR-100), top confusions, dead-channel fraction per stage,
accepted programs so far, and the last `--history` attempts with code and results
or tracebacks.

## Output: `runs/<timestamp>/`

| File | Contents |
|---|---|
| `best.pt`, `best_metrics.json` | Best weights and their val diagnostics |
| `baseline_val.json` | Val diagnostics of the start weights |
| `lineage.json` | Accepted programs in order (replay from start to reproduce `best.pt`) |
| `round_NNN/` | `input.json` (exact prompt), `program.py`, `result.json` (outcome, tokens) |
| `subset_indices.json`, `data_subset.pt` | The exact training data the programs got |
| `frozen.json`, `final_test.json` | Hash of `best.pt`; start vs best test accuracy |
