# Regime `reasoned`: images and labels meet only in the agent's reasoning

**In one line:** the agent sees the split and may analyse it, labels included. But every weight must be a constant or
formula written in a self-contained `model.py` and justified in its plan; no number computed from the data may be copied
into the network.

```bash
python -m ripple_wsp run reasoned --data /data/cifar10 --runs /data/wsp_runs
```

Exact first message: [../prompts/reasoned.md](../prompts/reasoned.md). Always starts from scratch (no `--seed`).

## What changes from `data`

| | `data` | `reasoned` |
|---|---|---|
| analysis code | may read the split and labels | same |
| weights | constants, formulas, or values measured on the split | constants or formulas written in `model.py` and justified in the plan; no copied numbers |
| `weights/` files | up to 100 MB, loaded by `model.py` | none: `model.py` loads no files |
| `model.py` size | unlimited (within 25M numbers) | at most 60 KB of source |
| copy check | none | rejected if 6 or more of its numbers match values the analysis printed |

Rule 2 as the agent reads it:

> Images and labels meet only in your reasoning. Your code may analyse the split, labels included, and print what it
> finds, and you decide the weights from what you learn, but no number computed from the data may be copied into the
> network. Every weight is a constant or a formula that you write in model.py and justify in your plan. Do not transcribe
> measured statistics (class means, centroids, prototypes, variances, per-class thresholds or histograms) into weights: a
> number you take from your analysis must be a small summary that you read and choose, with the reason stated. Linear
> solvers are disabled.

## How it is enforced

On top of everything in [data](data.md):

- **Self-contained `model.py`**: static scan rejects `open(`, `np.load`/`torch.load`/`fromfile`/`loadtxt`/`memmap`,
  `pickle`, `pathlib`, `shutil`, `importlib`, `exec`, `eval`, and importing `os`, `sys`, `io`, `glob`. The build box for
  this regime contains only `model.py`; no `weights/` directory is copied in.
- **60 KB limit** on `model.py` source.
- **Copy detector** (`harness.copied_numbers`): it collects every number printed by earlier turns' analyses, minus numbers that
  appear in the agent's own earlier `model.py` files. Each number in the new `model.py` with at least 3 significant digits
  that matches a printed value within its written precision counts as a hit. Six or more hits reject the file.

## Known gap: quantised copies pass the copy detector

We audited five 15-turn runs where Claude Opus played the agent through the [file backend](../backends.md). All five broke
rule 2 from turn 2 or 3, and the copy detector never fired:

- The analysis printed per-class tables (each feature's z-score or mean by class).
- `model.py` then held the same tables **rounded to a few levels** (±0.5, ±1, or {0.3, 0.5, 0.7, 1.0}), as nearest-class-mean
  or naive-Bayes readouts. 82/84, 550/550 and 221/230 entries matched the rounded printed values in three of the runs.
- The plans' visual explanations came after the tables and did not change any value.

| Agent | Final test (5 runs) | Notes |
|---|---|---|
| Ripple-65B (OpenRouter) | 13.0, 20.1, 16.5, 14.6, 17.7% | best turn by val: 18.8, 20.1, 22.7, 14.6, 20.0% |
| Claude Opus (file backend) | 37.4, 38.5% (2 runs finished) | **non-compliant** from turn 2: quantised class-statistic tables; the compliant turn-1 networks (written from priors before any data) scored 14 to 20% on the split |

Treat `reasoned` scores above roughly 20% as suspect until audited. To audit a run, compare each class row of the readout
in `model.py` with the per-class tables in the earlier `analysis.log` files: matching signs and rounded magnitudes give
it away. Fixes we have not implemented yet:

1. Let labels reach the agent only through evaluations of whole networks it wrote (accuracy and confusion matrix, 5 per
   turn), and stop analysis code from printing label-conditioned statistics.
2. Extend the copy detector to coarse matches: the sign pattern of a readout row equal to a printed per-class row, or
   values equal after rounding to one significant figure.
3. State in the task that rounded, quantised, coded or scaled versions of per-class statistics count as copies. During
   our runs one agent got this as a side message mid-run; it is not in the task text.
