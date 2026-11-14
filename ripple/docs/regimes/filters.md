# Regime `filters`: a generation that engineers the feature layers

**In one line:** a [generation](generation.md) whose task adds a note asking the agent to design the convolutional
layers for the classes its seed confuses, rather than only re-measuring the readout.

```bash
python -m ripple_wsp run filters --seed /data/saved/gen2 --data /data/cifar10 --runs /data/wsp_runs
```

Exact first message: [../prompts/filters.md](../prompts/filters.md).

## What changes from `generation`

The first message gets [`ripple_wsp/focus/filters.md`](../../ripple_wsp/focus/filters.md) inserted before
"You have 15 turns":

> This generation's focus is the feature layers. In the network above, the convolution is a generic filter bank that is
> the same for every task, and the classes are told apart only by the measured readout: nothing in the feature layers was
> designed for these ten classes. Engineer them. Use the split to find what separates the classes the network confuses
> most (its confusion matrix shows which), decide what visual cue would tell each pair apart, and write filters and layers
> that compute those cues. You may add layers, combine responses from earlier layers, and remove or replace filters that
> contribute little. For every filter or layer you add or change, state in your plan which cue it computes and which
> classes it is meant to separate, and check on the split whether it does.

The note was written for our generation-2 seed (a generic Gabor bank with a measured readout). If your seed's features
are already class-specific, edit the note's first two sentences, or add another note under `focus/` and a regime for it
in `regimes.py`.

Rules, enforcement, feedback and limits are exactly [data.md](data.md). `--seed` is required.

## Our results (Ripple-65B via OpenRouter, 5 runs, seed = generation 2, 42.4% val / 42.5% test)

No run beat its seed on val. Several agents added class-specific filters, but the readout re-measured on the bigger
feature set gained on the split, not on held-out images. For example, one run reached 46.0% split with 4.1M numbers
and scored 42.3% val / 42.4% test. Three of the five runs died when agent feature caches filled the shared disk; the
2 GB workspace cap was added after that.
