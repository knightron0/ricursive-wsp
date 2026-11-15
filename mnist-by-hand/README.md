# mnist-by-hand: an MNIST classifier whose weights were written, not trained

A weight-space-programming experiment on MNIST, done interactively with Claude Code (Opus 5.5) rather than
through the `wsp` harness. Every weight is a stroke Claude drew, a fixed filter or a small integer table it
typed. **No training code was written:** no gradients, no closed-form fits, and no automatic search over
weights. Code ran forward passes and printed results, and every weight change was a hand edit.

**Result: 93.13% on the 60,000 training images.** The final model has **not** been scored on the test set.
`combine.py` and `strokenet.py` refuse to run on test without `--final`.

## Rules we followed

| Allowed | Not allowed |
|---|---|
| Forward passes of a hand-written network on MNIST train | Training code of any kind (gradients, least squares, automatic tuning) |
| Looking at training images and misclassified examples | Copying data into the weights (e.g. class-mean images or stored examples) |
| Typing weights by hand after reading printed results | Using the test set before the final evaluation |

This is closest to the harness's `full` mode: training images and labels are visible and there are no gradients.
One difference: four kinds of choices were **calibrated by eye on training data**:
- the hole and bay penalty tables, read off printed per-class frequencies;
- β and γ, the strengths of those two tables;
- the inhibitory-stroke strength;
- the 10 class offsets.

That's a manual fit of a small number of values. The test score may come in somewhat below the training score.

## Progress (training-set accuracy)

| Version | What changed | Train acc. |
|---|---|---|
| v0 | 7×7 grids drawn with no data, then one linear layer (`eval_templates.py`; **test: 32.00%**) | – |
| v1 | Digits drawn as pen strokes; slant and width variants; blur; cosine matching | 57.2% |
| v2 | Redrew digits where MNIST actually puts them (centring by centre of mass shifts 6, 7, 9 and 2) | 68.2% |
| v3 | ±2 px shifts and two heights per drawing | 81.3% |
| v4 | Feature layer: 8 oriented line detectors, ReLU, 2×2 max-pool | 82.1% |
| v5 | Hole detector (fixed recurrent flood fill) plus a hand-set penalty table | 88.3% |
| v6 | New drawings: strongly slanted 1, curly 2 with a big bottom loop, slanted 8 | 90.5% |
| v7 | Bay (concavity) detector (fixed ray casts) plus a hand-set penalty table | 92.5% |
| v8 | Hand-set per-class offsets | **93.1%** |

Final per-class training accuracy, digits 0–9: 97.4, 97.0, 94.3, 90.9, 93.5, 90.1, 94.4, 96.1, 88.9, 87.8%.
The largest remaining errors are 8→9 (309), 9→0 (238) and 5→3 (215).

## The final model

Input: a 28×28 image, pixels in [0, 1]. Coordinates: x is the column, y is the row, and y grows downward.

1. **Fixed feature layer** (`strokenet.features`):
   - 8 bar-shaped 7×7 line detectors, one every 22.5°. Each is an elongated Gaussian (σ 2.0 along the line,
     0.8 across) minus a wider one, so it responds only to lines.
   - ReLU, then a 2×2 max-pool, then a light blur. Output: φ(x), with 8×14×14 = 1,568 values.
2. **Template layer** (`strokenet.DIGITS`):
   - 14 digit drawings made of lines and elliptical arcs. Some include inhibitory strokes, where ink counts
     against the digit (e.g. a 9 must have no lower-left loop, or it may be an 8).
   - Each drawing is rendered with a soft pen at 1,000 variants (5 slants × 4 widths × 2 heights × 25 shifts)
     and passed through the **same** φ. That gives 14,000 templates.
   - Templates are made zero-mean and unit-length, after removing half of the component they all share.
   - A digit's score is its best cosine match over its own templates.
3. **Shape detectors** (`holes.py`), on the image binarised at 0.3:
   - **Holes:** flood-fill the background from the border (40 steps of a 4-connected spread). Whatever the
     fill never reaches is a hole. Two yes/no features: a hole in the top band (rows < 12) and in the bottom
     band (rows ≥ 16).
   - **Bays:** cast rays left, right, up and down from every background pixel. A pixel blocked on exactly
     three sides is in a bay that opens toward the fourth. Eight yes/no features: open side (L, R, U, D) ×
     half (top or bottom).
4. **Output** (`combine.py`):
   `score_k = template_k + 0.05·hole_penalty_k + 0.03·bay_penalty_k + offset_k`, then argmax.
   - Penalty rule: −2 if that observation occurs in ≤ 5% of the digit's training images, −1 if in 5–25%,
     else 0.
   - Offsets for digits 0–9: −0.015, 0, −0.02, +0.03, 0, 0, +0.005, +0.015, −0.01, −0.01.

All weights are fully determined by the constants at the top of `strokenet.py` and `combine.py`.

## Run

```bash
pip install -r requirements.txt     # numpy only; MNIST downloads itself into data/ on first use
python combine.py train             # final model on train: 93.13% (~1 min, ~1.1 GB peak RAM)
python strokenet.py train           # template layer alone
python eval_templates.py            # v0 on test: 32.00%
python cmp.py 6,9                   # print my drawings next to random training examples of those digits
python combine.py test --final      # the one-time test evaluation (not run yet)
```

## Files

| File | Role |
|---|---|
| `strokenet.py` | Digit drawings, renderer, oriented feature layer, template matching |
| `holes.py` | Hole detector (flood fill) and bay detector (ray casts) |
| `combine.py` | Hand-set penalty tables, class offsets, final score |
| `mn.py` | MNIST download, loading, and ASCII viewer |
| `cmp.py` | Shows my drawings next to real digits |
| `eval_templates.py` | v0, the first no-data attempt |
