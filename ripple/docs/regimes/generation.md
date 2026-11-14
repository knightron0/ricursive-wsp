# Regime `generation`: improve an earlier network

**In one line:** the `data` regime, but turn 0 is an earlier network (its `model.py` and `weights/`), and the agent is told
to build on it.

```bash
python -m ripple_wsp run generation --seed /data/saved/gen1 --data /data/cifar10 --runs /data/wsp_runs
```

Exact first message: [../prompts/generation.md](../prompts/generation.md).

## What changes from `data`

- `--seed DIR` must hold `model.py`, plus `weights/` if `model.py` loads files. A finished turn directory
  (`RUN/turns/turn_XX/`) or a directory saved by `generations` works as is.
- Before turn 1, the harness builds and scores the seed on the split (turn 0). If the seed fails, the run stops.
- The seed's `model.py` is copied into the workspace, together with `weights/` and `history/turn_00_model.py`. The first message then gets
  this paragraph before "You have 15 turns":
  > You do not start from scratch. An earlier agent working under these same rules built the network below; it is your
  > submission now (turn 0), and its weights files are already in your workspace under weights/. Build on it and improve it.

  It is followed by the seed's full `model.py` and its split score (size, layers, accuracy, per-class accuracy, confusion matrix).
- The seed is the submission until a turn's `model.py` is accepted, and it counts as the best by split from the start.
  After the run, turn 0 is scored on val and test like every other turn, so `results.json` shows whether the agent beat
  its seed.

Everything else (rules, enforcement, feedback, limits) is exactly [data.md](data.md).

## Repeated generations

```bash
python -m ripple_wsp generations generation --seed /data/saved/gen1 --saved /data/saved \
    --data /data/cifar10 --runs /data/wsp_runs --generations 5 --replicates 3
```

Each generation starts `--replicates` runs from the current seed as parallel subprocesses and waits for them. Then it
takes every accepted turn of those runs, picks the one with the **highest val accuracy**, and makes it the next seed if it
beats the current seed's val. The new seed is saved as `SAVED/gen<N>_val<acc>_<time>/` (`model.py`, `net.pt`, `weights/`,
`README.txt`). Decisions are logged to `SAVED/generations.jsonl`. Arguments after `--extra` go to every `run`, for example
`--extra --turns 10`.

Select by val, never by test. Test is for the one number you report at the end.

## Our results (Ripple-65B via OpenRouter)

| Generation | Seed | Best network | Split | Test | Size |
|---|---|---|---|---|---|
| 1 | none ([data](data.md)) | 12 hand-written filters, measured readout | 39.2% | 37.9% | 8,026 |
| 2 | generation 1 | 1,036 formula filters (Gabor/Gaussian x colour), max-pool, BatchNorm + Linear class-mean readout | 44.7% | 42.5% | 1,305,371 |

Caveats from our own audit of generation 2:
- About 61% of its numbers are the measured readout. The 1,036 filters are a generic Gabor/Gaussian bank written by
  formula, not designed for these classes. That is why the [filters](filters.md) regime exists.
- One generation-2 analysis compared more than 5 variants by computing accuracy itself. Rule 4 is enforced
  mechanically only for `lab.evaluate`.
- We first ran generations 3 to 5 with seeds picked by **test** accuracy. Do not report those numbers. The `generations`
  command here picks by val.
