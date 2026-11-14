# Regime `generalize`: engineer filters and judge designs on held-out split images

**In one line:** the [filters](filters.md) regime plus a second note telling the agent that its split accuracy overstates
accuracy on new images. It must set part of the split aside, measure weights on the rest, and judge designs on the
held-out part. The split stays the only data.

```bash
python -m ripple_wsp run generalize --seed /data/saved/gen2 --data /data/cifar10 --runs /data/wsp_runs
```

Exact first message: [../prompts/generalize.md](../prompts/generalize.md).

## What changes from `filters`

After the filters note, the first message also gets [`ripple_wsp/focus/generalize.md`](../../ripple_wsp/focus/generalize.md):

> Your network is judged only on images it has never seen. The split accuracy I report each turn is measured on all
> 5,000 images, including the ones your measured weights came from, so it overstates accuracy on new images, and the more
> numbers you measure from the split, the more it overstates. Aim for out-of-sample accuracy: when you measure weights,
> set part of the split aside, measure on the rest, and check on the part you set aside; report both numbers in your
> plan, and prefer designs that hold up on the set-aside part over ones that only score higher on the images they were
> measured on. Checks on the set-aside part count toward your 5 variants per turn. The split is still the only data you
> may use.

The harness does not change what it reports: split accuracy is still on all 5,000 images. The held-out part is the
agent's own bookkeeping, and checks on it count toward the 5 variants. The data boundary is unchanged: Landlock still
gives agent code only `sample.npz`, and val and test stay invisible.

Rules, enforcement and limits are exactly [data.md](data.md). `--seed` is required.

## Our results (Ripple-65B via OpenRouter, 5 runs, seed = generation 2, 42.4% val / 42.5% test)

One run beat the seed: its turn 10 (Tanh nonlinearity, chosen on its own held-out part) scored 42.9% val / 42.9% test
against the seed's 42.4% / 42.5%. The other runs ended at or below the seed, and one collapsed to 11% by its final turn.
This is the same seed and val selection as the [filters](filters.md) arm, so the two regimes can be compared directly.
