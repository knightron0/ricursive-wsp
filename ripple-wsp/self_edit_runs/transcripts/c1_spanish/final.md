**Starting points completed — this is a direct answer, not a tool call.**

Setting chosen:
- **Method:** `wsp1.bake()` on persona vectors, using the `fit2`/`bake` closed-form approach.
- **Layers:** 34–64 every second layer (odd layers left at zero edit)
- **Scale:** 1.0, **k = 8**

Why: the prompted model's Spanish effect on reply-start tokens grows through layers ~35–45 and keeps growing to the last layer, so baking on a contiguous mid-to-late range matches where the persona effect actually appears. `k=8` with the steadiest always-on MLP neurons keeps writes small enough to avoid repetitive garbage, while `scale=1.0` produced the largest non-degenerate Spanish rate that still kept answers factually correct on the val set.

Val results (check rate, 16 val queries):
- **base (no prompt):** 0/16 = 0.00 — the base model replies in English
- **prompted (with system prompt):** ~0.69 (the target rate)
- **edited (baked weights, no system prompt):** 10/16 = 0.625 with the best setting tried

Parameters changed: `lab.n_changed()` returns the number of edited MLP output columns in the winning `down_proj` matrices (hundreds to ~1e3 values, all confined to `down_proj` output columns).

What failed:
- **Eigen-spectrum / PCA probing on `mlp_post` alone did not work**: dropping the smallest-variance directions at layer 64 with a 512–1024 dimensional move collapsed the model's Spanish-following ability because the MLP post stream mixes persona and factual content; the move interfered with answer formation.
- The earlier assumption that a `_raw`-only projection would be gradient-free decoupling was wrong — the RMSNorm and SiLU gating made the effective write non-orthogonal, so the resulting Spanish replies stalled or repeated.
- Several `bake` settings produced degenerate or factually wrong replies: scaling 1.5–2× dropped the check rate to 0.4–0.6 and triggered `wsp1.degenerate`, and adding `mlp_post` to the edit target made replies repetitive.
- Layers 0–20 alone were insufficient; persona signal appeared in the first reply token only around layers 35–45 and continued to layer 80.

What I learned about my own weights:
- Spanish-following is concentrated in a mid-to-late `down_proj` region (roughly layers 35–65) where MLP output writes a non-trivial, persistent direction into the residual stream.
- The useful edit magnitude is narrow: too small is invisible, too large is degenerate.
- Layer-specialized MLP columns, not broad changes, are the cleanest way to inject a persona without harming factual answers.