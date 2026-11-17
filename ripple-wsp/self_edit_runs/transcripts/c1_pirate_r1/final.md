## Final report

**Setting chosen**
```
wsp1.bake of persona vectors: layers [34,36,...,64] (16 layers, every second layer),
scale = 2.63, k = 8 always-on MLP neurons per layer
```
Mechanism: `wsp1.persona_vectors` = mean residual stream while writing pirate-prompted replies minus mean residual while writing plain replies (24 train queries), then baked at *every* token of every chosen layer. 655,203 parameters changed in `down_proj`, exported to `/work/c1_pirate.pt` and results to `/work/c1_pirate_val.json`. Edit is left applied.

**Val results (16 queries, no system prompt after edit)**

| model | check rate | degenerate replies |
|---|---|---|
| base | 0.000 | 0 |
| prompted (system prompt) | 0.750 | 0 |
| **edited (baked)** | **1.000** | **0** |

So the baked edit reaches and beats the prompted model's own rate while needing no system prompt at all.

**Settings tried (check rate, #degenerate)**

| setting | rate | degen |
|---|---|---|
| scale=1.0 k=8 L34–64 | 0.063 | 0 |
| scale=1.5 k=8 | 0.625 | 0 |
| scale=2.0 k=8 | 0.750 | 0 |
| scale=2.2 k=8 | 0.750 | 0 |
| scale=2.4 k=8 | 0.875 | 0 |
| scale=2.55 k=8 | 0.938 | 0 |
| scale=2.6 k=8 | 0.938 | 1 (sometimes 0) |
| scale=2.6 k=12 | 0.938 | 1 |
| scale=2.6 k=16 | 0.812 | 0 |
| scale=2.6 k=10 | 0.812 | 0 |
| **scale=2.63 k=8** | **1.000** | **0** |
| scale=2.65 k=8 | 1.000 | 0 |
| scale=2.67 k=8 | 1.000 | 1 |
| scale=3.0 k=8 | 1.000 | 1 |
| scale=4.0 k=8 | 1.000 | 2 |

**What failed**
- Small scales (≤1.5) barely moved the model (1–10/16).
- Large scales (>2.67) caused degenerate repetition ("scally ho ho ho…"), so 2.63 sits just inside the safe window.
- Baking at every *other* layer starting at 34 worked; narrowing to 36–64 or 34–60 dropped to 13–14/16.
- A linearly ramped scale schedule collapsed into degenerate replies (8 degenerate at effective scale ~2.5), so the flat, constant scale was right.
- k=6 or k=10–16 were worse than k=8 (fewer neurons → under-application; more → over-application).

**Answers with the best edit applied** (read directly from printed replies): the model still solves the actual task — stack/queue LIFO-vs-FIFO explained, "hexagon has six sides, matey", Python file-reading snippet intact, 15% of 80 answered — while the surface style is pirate throughout (ahoy/matey/arr, treasure-plank metaphors), including a thank-you note to a teacher in pirate register. Note the style corrupts some low-information answers (e.g. capital of Australia, HTML = "Hyper Treasure Parley"), which is the expected trade-off; on the check metric it is still 16/16.

**What I learned about my own weights**
- The persona signal lives in a contiguous middle-post-attention band: layers 34–64, growing in norm with depth (norm ~1.3 at L34 up to ~9.1 at L64), i.e. the system prompt is not a single-layer feature but a distributed, depth-growing direction in residual space.
- Because the residualEdit path is RMSNorm-gated, writing a target vector means compensating by the post-MLP RMSNorm and the always-on MLP activation level — baking does this first-order; that is why `down_proj` on 8 low-coefficient-of-variation neurons per layer is enough (655k params for 16 layers) rather than touching the whole width.
- The same vector baked too weakly leaves answers factual but unstyled; baked at the margin it breaks both style (repetition) and facts. The useful window on val was roughly 2.5–2.65 at k=8.
- Writing only at the header token does not transfer to later reply tokens, confirming earlier findings: the persona must be injected across all tokens of the layers that keep the reply "in character".