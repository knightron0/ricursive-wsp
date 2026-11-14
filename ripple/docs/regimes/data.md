# Regime `data`: the split, measurement allowed, no fitting

**In one line:** the agent sees the 5,000-image split and writes a network whose weights are constants, formulas, or
values its code measures directly on the split, never values that minimise a loss.

```bash
python -m ripple_wsp run data --data /data/cifar10 --runs /data/wsp_runs
```

Exact first message: [../prompts/data.md](../prompts/data.md). The `generation`, `filters` and `generalize` regimes use the
same rules and differ only in the starting network and an extra note.

## What the agent gets

- **The split** through `import lab`: `lab.load_split()` returns `x` uint8 [5000, 32, 32, 3] (HWC, RGB) and `y` int64
  [5000]; `lab.CLASSES`; `lab.prepare(x)` gives the float tensor [N, 3, 32, 32] in [0, 1] that the network receives;
  `lab.evaluate(net)` returns a network's accuracy on the split, at most 5 times per turn.
- **A workspace**: a directory that persists across turns (files the agent's code writes stay), with earlier turns' code
  under `history/` and the current `model.py`. Capped at 2 GB.
- **Compute**: numpy and PyTorch, CPU only, 16 threads, 10 minutes and 64 GiB per analysis run, no network, no GPU.
- **Nothing else**: no val, test, other train images, other runs, internet or environment variables.

## The rules (as the task states them)

1. No training: no gradients or backward passes, no optimizers, no finite-difference or perturbation estimates, no loop
   that adjusts weights from a loss or accuracy (no random, evolutionary or grid search over weights).
2. Weights are constants or formulas the agent writes, or values its code computes from the split by direct measurement
   (for example class means, filter statistics, normalisation constants). They may not come from minimising a loss or an
   error, iteratively or in closed form. Linear solvers are disabled.
3. The model is a network: `build()` returns a `torch.nn.Sequential` of Conv2d, Linear, ReLU, LeakyReLU, Tanh, Sigmoid,
   MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d, Flatten, BatchNorm1d, BatchNorm2d, Softmax, Identity and
   nested Sequential. Input [N, 3, 32, 32], output [N, 10], argmax is the label. At most 25,000,000 numbers; 10,000
   images in under 5 minutes on CPU. A nearest-centroid or prototype classifier is allowed only if expressed with these
   layers (for example a Linear layer whose rows are class means).
4. At most 5 variants per turn: `lab.evaluate` works at most 5 times, and the agent must not compare more variants by
   other means.

`model.py` may load arrays the agent's code saved under `weights/` in the workspace (at most 100 MB); it reads nothing else.

## How the rules are enforced

| Rule | Mechanism | Strength |
|---|---|---|
| no gradients, optimizers | `guard.lock_torch`: grad mode off and cannot be turned on; `backward`, `torch.autograd.*`, `torch.func` transforms and every `torch.optim` optimizer raise. Static scan of analysis code and `model.py` rejects them before running. | mechanical |
| no linear solvers | `guard.lock_solvers`: `np.linalg.{solve,lstsq,pinv,inv,...}`, `torch.linalg.*` solvers, `torch.inverse/pinverse/lstsq` raise. Static scan too. | mechanical |
| no search over weights, no iterative fitting without solvers | the 5-evaluate limit per turn; everything else is instruction only | **instruction** (audit the transcripts) |
| only the split | Landlock: agent code reads only the Python install, `ripple/` and `sample.npz`; writes only its workspace; no TCP. Static scan rejects absolute paths, `..`, `os.environ`, subprocesses, network packages. | mechanical |
| network of allowed layers only | `model.py` is built in an empty box (only `model.py` and `weights/`), checked layer by layer (exact types, no overridden `forward`, no hooks), saved, then scored in a fresh process with `torch.load(weights_only=True)`, so no agent code runs at scoring time. `model.py` may not import `lab`. | mechanical |
| at most 5 variants | `lab.evaluate` raises after 5 calls; every `load_split` and `evaluate` is logged through a file descriptor the harness owns and reported back to the agent | mechanical for `evaluate`, instruction for other comparisons |

## Each turn

The reply has a plan, optionally one ```` ```python ```` analysis block, and exactly one block whose first line is
`# model.py` (the complete file, every turn, even if unchanged). Then:

1. **Static scans.** Rejected code is not run, and the reply says why.
2. **Analysis.** It runs in the workspace under the sandbox, and the end of its output (up to 5,000 characters) goes back to the agent.
3. **`model.py`.** It is built and scored on the split. If it is accepted it becomes the submission. If it is missing or fails, the previous
   submission stays. If the block is missing, the harness asks once for just that block.
4. **Feedback.** The agent gets the analysis output, the split reads this turn (`load_split` count, `evaluate` count and
   accuracies), the network's size and layers, split accuracy, per-class accuracy, the full confusion matrix, the
   best split accuracy so far and the turns left.

Older turns are shortened in the context (plan plus short feedback; code stays in `history/`) to fit 80,000 tokens.

## After the run

Every accepted turn's `net.pt` is scored on `val.npz` and `test.npz`. `results.json` records the **final submission**
(the last accepted network), the **best by split** and the **best by val**. Report the final submission, or pick by val.
Never pick by test.

## Our results (Ripple-65B via OpenRouter)

| Runs | Best network | Split | Test | Size |
|---|---|---|---|---|
| 3 runs from scratch (first generation) | 12 hand-written filters, measured class-mean readout | 39.2% | 37.9% | 8,026 numbers |

Split accuracy well above test means memorisation: one network built from measurements scored 96.7% on the split and
18.2% on test. Always read split and test together.

Caveats from our own audit:
- The readout layer of the best networks is a direct measurement (class means of features), which these rules allow but
  which is most of the learned capacity. Ridge regression on raw pixels gets 37.6%, so treat ~38% as "measured readout
  on hand-designed features", not as knowledge-only programming.
- Rule 4 beyond `lab.evaluate` is instruction only; check analysis code for loops that compare many variants.

The task text in our first-generation runs lacked the 2 GB workspace sentence, which was added later after agents filled
a shared disk with feature caches. Otherwise it is identical to [../prompts/data.md](../prompts/data.md).
