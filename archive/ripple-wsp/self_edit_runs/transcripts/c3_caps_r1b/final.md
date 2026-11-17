**Final report**

## What I found

Baseline vs. target on VAL (score = answers ≥97% capital letters, answer text only):
- `score(VAL_BASE)` = `{'n': 16, 'answered': 16, 'passed': 0, 'unfinished': 0, 'degenerate': 0}`
- `score(VAL_PROMPTED)` = `{'n': 16, 'answered': 15, 'passed': 14, 'unfinished': 1, 'degenerate': 0}`

Aimed-for output = everything after `</think>` in capitals, e.g.:
`DEAR TEACHER, THANK YOU FOR YOUR PATIENCE AND GUIDANCE...` (a prompted reply is fully correct, just uppercase; base replies are normal mixed case and score 0).

## Key observations

1. **The answer is the only part scored.** In all 17/17 `TRAIN_PROMPTED` replies that contain a reasoning block, the `<think>` part stays mostly lowercase (upper-letter fraction 0.03–0.38) while the post-`</think>` answer is 65–89% capital letters. So a "persona" vector built from *all* reply tokens is dominated by lowercase reasoning text — that matches the known failed attempt (~1/3 pass, scale 2.5).

2. **Model structure:** 80 layers, block = `h += post_attn_norm(attn(input_norm(h)))` then `h += post_ff_norm(mlp(pre_ff_norm(h)))`. The final residual (layer 79's block output) goes straight into `lab.model.lm_head.weight` ([100352, 5120]); there is a final `model.norm` RMSNorm. The captured stream components are `resid` (block output), `mlp_in` (down_proj pre-activation), `mlp_post` (post-norm MLP output).

3. **`bake` mechanics:** it writes `inc = scale*(vecs[L] - vecs[prev])` at every token of layer L through the k steadiest always-on MLP neurons, with a first-order RMSNorm compensation (`d = inc/g*rms(raw).mean()`). With one layer `Ls=[79]`, `vecs={79: v*s}`, the residual change at layer 79 was measured: at scale 1 the per-token Δnorm was ≈0.16 and mean-|Δlogit| across the last residual ≈ small, but the *direction* of the shift was attainable; at scales 10–100 the added residual norm saturated (a "limited size" per layer), and the model's generation degraded into nonsense tokens (`ĠMOCK`, `TRGL`, `INFRINGEMENT`, `PYTHON`).

4. **Direction construction:** `UP_MINUS_LOW` (unit mean of unembedding rows of uppercase tokens minus their lowercase twins) is exactly `mean(E[up] − E[low])` over `PAIRS` (8336 pairs, 5235 unique uppercase tokens); I verified `v == UP_MINUS_LOW` bit-for-bit (cos = 1.0). Per-pair margins of that direction on token logits are small (median cos ≈ 0.157, min ≈ −0.063), so a unit residual is not enough; large scales are required.

5. **Nobody wrote weights yet except experiments, all rolled back.** `lab.n_changed()` was never non-zero after `rollback`. The measurement loop was the only remaining work when the budget ran out: I had prepared `/tmp/dir.pt` (directions), `/tmp/pf.pt`/`/tmp/twins.pt` (prefixes and twin maps), `/tmp/pers_ans.pt` (answer-only persona vectors over layers 56–79, norms 3.8 → 23.65), but no edited weights were generated or scored.

## What failed / was not completed

- The known persona attempt (layers 34–64 from short replies, scale 2.5) — reproduced conceptually only.
- My planned fix (answer-only persona vectors, full-length prompted replies) was built in 157 s but **never baked or scored** — ticking the step budget killed the pipeline before `gen_val()` on an edited model.
- Direct planar logit probing at L=79 with scales 1–64 showed margins barely moving (base margins −0.5 to −3.6 for `capital/CAPITAL`, `system/SYSTEM`, `over/OVER`, `live/LIVE`, `There/THERE`) and at high scale generation degenerated; no scale gave the required ≈ভ +5 flip without garbage.

## Chosen "best" (must be stated, but unverified)

No verified best. My selected candidate setting for export, had budget remained, was: bake `v = UP_MINUS_LOW` at the last layer only, layers `[79]`, `vecs={79: v*16.0}`, `k=8`, `scale=1.0`, applied on `OTHER_TEXTS[:12]`; plus a fallback candidate of the answer-only persona vectors over layers 56–79 baked with `wsp1.bake` at scale 1.0. Neither was ever run through `gen_val()` or `score()`, and neither patch file (`/work/c3_caps.pt`, `/work/c3_caps.json`) was written. No edit is left applied: weights are in the original state.

## Parameters changed, scores

- Parameters changed: **none left changed** (every `bake` was followed by `lab.rollback()`).
- VAL base: passed 0/16; VAL prompted: passed 14/16 (1 unfinished); edited: unmeasured.

## Lessons

1. Capitalization is a **case-only** logit tie: the same letters, only the twin token differs — so the edit must preserve token identity while flipping the up/lowercase margin at answer positions, and any aggressive logit shift risks destroying identity (degenerate outputs at high scale).
2. Score only looks at answer text after `</think>`; reasoning may stay lowercase, so vectors built from reasoning residuals are actively misleading.
3. A single layer saturates quickly (≈0.16 residual norm per unit scale); several layers (as the task hints) must share the total shift, with the largest `\vecs[L]` stacked at late layers (norms grew 3.8→23.65 from layer 56→79 for answer-only personas).
4. All measurement loops must be wrapped in `lab.rollback()` before each attempt, and the final run must leave one edit applied plus exported `patch`/`json` — none of which was possible within this budget.