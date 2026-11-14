# Regime `blind`: no data at all

**In one line:** the agent gets the official CIFAR-10 description and nothing else: no images, no seed, no accuracy
feedback. It writes every weight by hand or by formula from what it knows about small photos of these ten classes.

```bash
python -m ripple_wsp run blind --data /data/cifar10 --runs /data/wsp_runs
```

`--data` is still needed because the final network is scored on `val.npz` and `test.npz` after the last turn. The agent
never touches either. Exact first message: [../prompts/blind.md](../prompts/blind.md).

## What the agent gets

- **The dataset description**, quoted from the CIFAR-10 page: 60,000 32x32 colour images in 10 classes, 6,000 per
  class, 50,000 train and 10,000 test, the test batch has exactly 1,000 per class, the classes are mutually exclusive
  (automobile vs truck, no pickup trucks). It also gets the class names in label order.
- **A workspace and compute** as in [data](data.md), but `import lab` gives only `lab.CLASSES` and `lab.prepare(x)`.
  `lab.load_split()` and `lab.evaluate()` raise "this task has no data".
- **Synthetic images**: the agent may draw its own images (numpy arrays) and run its network on them, to check that a
  filter responds the way it intends.

## The rules (as the task states them)

1. No training (as in [data](data.md)).
2. No data. Every weight is a constant or a formula the agent writes. It must not write in numbers it remembers from
   this dataset or from networks trained on it (for example per-channel pixel means or trained weights). Linear solvers
   are disabled.
3. The model is a network (same layers and limits as [data](data.md)).
4. At most 5 variants per turn, compared on synthetic images it draws. Anything broader is search.

## How the rules are enforced

- **No images reachable**: Landlock gives agent code no data file at all. The harness does not even pass the split's
  path into the sandbox, and the blind `lab` module has no loader.
- **No feedback**: after each turn the harness only builds the network and checks that it maps [N, 3, 32, 32] to [N, 10].
  The feedback is "model.py accepted: it builds ... network: N numbers; layers: ... No accuracy is reported in this
  task." The split is never scored and no split reads are reported.
- **No training or solvers**: the same torch locks and static scans as [data](data.md).
- **No remembered statistics or weights**: instruction only. Read the plans and `model.py` for suspiciously precise
  constants.

## After the run

Every accepted turn is scored on val and test. With no feedback, "best by split" does not exist: `results.json` records the
final submission and the best by val. Report the final submission. Best-by-val is a post-hoc choice the agent could not
have made.

## Our results (Ripple-65B via OpenRouter, 5 runs, 3 finished)

| Run | Final submission (test) | Best turn by val (test) | Size of final network |
|---|---|---|---|
| 1 | 16.1% | same | 122 numbers |
| 2 | 13.7% | 15.2% | 881 |
| 3 | 10.2% | 11.7% | 5,510 |

Chance is 10%. Mid-run checks showed networks collapsing onto one or two classes: without feedback the agent cannot
calibrate the readout biases, so one class's score dominates. A hand-written 40-number colour network (blue top means
airplane or ship, green means frog, and so on) gets 17.3% test, in the same range.

Our blind runs used the task text from before the 2 GB workspace sentence was added. Otherwise it is identical to
[../prompts/blind.md](../prompts/blind.md).
