# Figures

This directory keeps one selected version of each figure used in the CIFAR-10
write-up. Drafts, alternate layouts, thumbnails, generated-image outputs, and
one-off rendering scripts are omitted.

| File | Content |
| --- | --- |
| [hero.png](figures/hero.png) | Overview of the three experiment settings |
| [reasoning-phases.png](figures/reasoning-phases.png) | Tools available before and after `end_tools()` |
| [label-recovery.png](figures/label-recovery.png) | Illustrative label recovery through confusion matrices |
| [opus-ascii.png](figures/opus-ascii.png) | Images and their ASCII representations |
| [opus-conv1-kernels.png](figures/opus-conv1-kernels.png) | Opus's first-layer filters |
| [opus-fc-votes.png](figures/opus-fc-votes.png) | Class votes from the detector features |
| [astra-synthetic.png](figures/astra-synthetic.png) | First grid of synthetic examples |
| [ripple-synthetic.png](figures/ripple-synthetic.png) | Pixel-art synthetic examples |
| [opus-blind-airplane.png](figures/opus-blind-airplane.png) | Hand-built filters and their airplane-score contributions |

These PNGs are presentation assets, not raw measurements or a substitute for
the `result.json` and `audit.json` emitted by the experiment harness. Full run
artifacts are created locally under the ignored `runs/` directory.
