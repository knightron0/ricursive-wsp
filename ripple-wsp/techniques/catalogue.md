# Weight-Space Programming (WSP) technique catalogue

Compiled 14 Nov 2026 (UTC) for the Ripple-65B "system prompt without a system prompt" task
(80 layers, d_model 5120, MLP width 40960, PeriNorm blocks with RMSNorm after each sublayer, gated attention,
value residuals, no biases; no gradient training allowed in the edit).

Raw fetched pages and cloned repos were not committed (third-party copyright); every entry links its source.

## How to read this

Evidence labels used in every entry:

- **[READ]** I read this in the source text or code this session.
- **[SKIM]** I only saw the abstract, headings or grep hits; details may be missing.
- **[MEMORY]** Classic method written from my memory of the paper; not re-fetched this session. Check before relying on constants.
- **[INFERRED]** My own reasoning, not stated by any source. All "Apply to Ripple" notes are [INFERRED]; nothing here was run on Ripple-65B.

Notation: `W_out[l]` = MLP down/output projection of layer l, shape (d_model, d_mlp); column j is the residual write
direction of neuron j. `O[l]` = attention output projection, shape (d_model, n_heads·d_head); head h owns columns
`h·d_head:(h+1)·d_head`. `N(x) = x/‖x‖`. `E_t` = embedding row of token t, `γ` = stored final-norm gain;
"effective readout" `r(t) = E_t ⊙ (1+γ)` (WSP-Scope's convention for Qwen3.5, whose RMSNorm stores gain−1 — check
Ripple's convention; with tied or untied unembedding use the unembedding row).

### Three architecture facts that change every recipe on Ripple-65B [INFERRED]

1. **No biases.** A steering vector cannot be added as a bias. The only published workaround is the
   *pseudo-bias* of T7: a vector `c` with `c·x ≈ 1` on every token (closed-form ridge of the constant 1 onto a
   layer's MLP inputs), used as an input row of a repurposed neuron. Verify that such a `c` exists in Ripple
   (it existed in Gemma 4 because of "massive" constant dimensions; `c·x = 1.000 ± 0.008`).
2. **RMSNorm after each sublayer.** Anything written through `W_out`/`O` is re-normalised before it reaches the
   residual stream, so (a) scaling a whole matrix does nothing, (b) an added direction competes with the sublayer's
   natural output for a fixed norm budget, so the per-layer shift is capped and edits should be spread over layers,
   and (c) a target residual shift `δ` must be mapped back through the post-norm gain: write `δ ⊘ gain_post` (zero
   the dimensions whose gain is tiny). T7 did exactly this on Gemma 4 ("mapped back through the post-feedforward
   RMSNorm weight and the layer scalar. Dimensions where the post-FF norm weight is below 0.1 are zeroed").
   Directional *removal* (T1) is exact only up to this re-normalisation.
3. **Gated attention / value residuals.** T6 is the only published edit of an attention gate (Qwen3.5-9B's
   `q_proj` interleaves query and gate rows per head). Value residuals mean an early-layer value edit leaks into
   all later layers; no source covers this.

For the task itself the natural "contrast" is always the same and is free of gradients: run the same user prompts
**with** the system prompt (S) and **without** (N), teacher-force the S-run's answer in both, and take differences
of residuals / MLP activations at assistant positions. Every recipe below is expressed against that contrast.

---

## A. Techniques published under "WSP" since 2 Nov 2026

### T1. Rank-one directional removal from output matrices ("loop direction"; abliteration-style)

- **Source** [READ]: Sparks of WSP, Experiment 8, https://rikiyat.github.io/blog/weight-space-programming/ ; code in
  `lm_understand-logs/exp8_math/runs/p50c/tool_calls.jsonl` and `runs/p10c/final_answer.md`
  (https://github.com/RikiyaT/lm_understand-logs).
- **What is edited**: `mlp.down_proj.weight` (and in run p10c also `self_attn.o_proj.weight`). p10c: 30 matrices in
  layers 8–22, dense. First p50c versions: whole `W_out` of layers 15 and 19.
- **Recipe**:
  1. One teacher-forced pass over matched pairs of texts: the model's own *productive* reasoning vs its own
     *repetitive* continuation (4 pairs in `fwd_4`, 6 in `fwd_24`). Save hidden states per layer.
  2. Per text, mean hidden state over the last 24–36 tokens; `d_l = mean(repetitive) − mean(productive)`;
     `u = N(d_{l+1})` (residual *after* layer l).
  3. `W' = W − α · u (uᵀ W)`. α=1 is exact projection; p10c kept α=1.5; p50c swept α ∈ {1,2,4,6}.
  ```python
  # exp8_math/runs/p50c/tool_calls.jsonl, code run 16
  meansH={l:np.stack([a[i,lens4[i]-36:lens4[i]].mean(0) for i in range(8)]) for l,a in H4.items()}
  diffH={l:(a[1::2]-a[::2]).mean(0) for l,a in meansH.items()}
  ...
  d=torch.tensor(diffH[min(l+1,23)]); d/=d.norm()
  w=downweights[l]
  new=w-strength*torch.outer(d,d@w)
  save_edit(f"model.language_model.layers.{l}.mlp.down_proj.weight",new)
  ```
- **Feedback**: 1 activation-read pass to get `u`; generation passes to pick α and layers (dose–response);
  a norm-matched random-direction control.
- **Result**: p10c (30 matrices, α=1.5): MATH-500 7.9→12.3 %, token-limit hits 93→87 %. L19-only α=1: relative
  change 3.4 % of the matrix, weak effect. The dense version was superseded by T2.
- **Failures**: p10c's own second holdout was neutral. Frontier-weight-edit's B1 baseline (same recipe, fixed
  sweep, on Gemma 4 12B) made the target behaviour *worse* (26→49 %) and broke 53 controls: "the naive ablation
  recipe breaks the past tenses altogether". Removal only works when the unwanted behaviour is carried by one
  direction that nothing else needs.
- **Apply to Ripple**: removal cannot add "write in uppercase"; use it with `α < 0` (amplify what the matrices
  already write along `u = N(mean_S − mean_N)`) or to remove a "lowercase/normal-register" direction. Weak first
  choice; cheap to try as a baseline.

### T2. Sparse top-K "writer neuron" directional edit (the headline Qwen edit)

- **Source** [READ]: same as T1; `runs/p50c/readable/notes.md`, `runs/p50c/edit/manifest.json`, code runs 61/90.
- **What is edited**: 64 columns of `W_out` in each of layers 15 and 19 of Qwen3.5-0.8B (1024×3584):
  2×64×1024 = 131,072 values (0.016 % of the model; 130,188 differ in bf16). Relative matrix change 10.6 % / 8.1 %.
- **Recipe** (final): `W = W0 − 6 · outer(u, mask64 ⊙ (uᵀ W0))`, where `mask64` selects the 64 neurons with the
  highest *writer score* `score_j = (uᵀ W0[:,j]) · (ā_j^rep − ā_j^prod)` (write projection × activation contrast).
  ```python
  # exp8_math/runs/p50c/tool_calls.jsonl, code run 61
  w=load_original(f"model.language_model.layers.{layer}.mlp.down_proj.weight")
  proj=u@w
  ...
  means=np.stack([np.asarray(a[i,n-24:n],dtype=np.float32).mean(0) for i,n in enumerate(meta["lengths"][:count])])
  scores.append(proj*torch.tensor((means[1::2]-means[::2]).mean(0)))
  score=scores[0]
  top=torch.argsort(score,descending=True)
  neuron_subsets[layer]=top[:64].clone()
  ```
  `u` is the unit repetition direction at the residual output of the layer (hidden 16 / 20), sign fixed so that
  repetitive − productive projects positively. With α=6 each selected neuron's write along `u` is multiplied by −5:
  the neurons now push *against* the direction (over-correction, not ablation).
- **Feedback**: 50 forward passes in total (activation reads + paired generations). Top-64 chosen on one read pass
  and checked for transfer on another (rank correlation, top-64 overlap).
- **Result**: MATH-500 7.9→48.1 %, GSM8K 13.8→67.1 %, IFEval 21.0→62.3 %, token-limit hits 93→33 %. Teacher-forced:
  repetitive branch −0.103 nats/token vs productive −0.009. Random norm-matched control +1.6 pp (SE 3.5).
- **Failures**: fixes a failure that hid an ability; on Gemma 3 270M (no loops) no edit moved MATH-500 by more than
  ~1 point (T4 entries). Remaining whitespace degeneration after the edit. HN/Goedecke/Honnibal: this is a
  steering-vector/abliteration edit found with forward passes, not weight reading [SKIM for Goedecke and HN].
- **Apply to Ripple**: same score with the S/N contrast finds the MLP neurons that already write the
  uppercase/pirate direction when the system prompt is present; then use α<0 (amplify) on those columns. They will
  only fire if their *inputs* fire without the system prompt, so pair with T7/T9 that supply the missing drive.

### T3. Attention-head knockout / rescale

- **Source** [READ]: Sparks p25c (`runs/p25c/final_answer.md`); WSP-Scope §7 (https://arxiv.org/html/2611.09382).
- **What is edited**: one head block of `o_proj`: `O[l][:, h·d_head:(h+1)·d_head] *= s`.
  Published settings: s=0 (Qwen-0.8B L15 H7, cols 1792:2048 — p25c and WSP-Scope MMLU-Pro; L15 H6 GPQA; L23 H5
  SciCode; 9B L23 H9 GPQA), s=0.5 (L11 H1, L15 H7, L23 H5 jointly, SciBench), s=2.5 (9B L23 H9, SciCode),
  s=1.1 (Gemma 3 270M L11 H2). 262,144 values per head at 0.8B, 1,048,576 at 9B.
- **How the head is found**: p25c: "localization identified this head's strong retrieval of repeated
  continuations. Weakening it suppressed loops dose-dependently, amplifying it reinforced them, and a same-layer
  control head did not reproduce its selective effect." I did not read the localisation code [SKIM].
- **Feedback**: attention patterns / activation reads plus paired generations.
- **Result**: p25c MATH-500 7.9→14.1 %; WSP-Scope held-out gains vary by task (e.g. GPQA +4.42 pp at 0.8B).
- **Failures**: WSP-Scope's editor kept the original weights for several 9B tasks (GPQA 261→259/360 in
  confirmation); three reruns of one search found different edits with +29.45 / +5.00 / +27.30 pp.
- **Apply to Ripple**: knockout cannot create a behaviour. Scaling *up* heads that copy case/register from
  earlier assistant tokens could make uppercase self-sustaining once started, but does not start it. Low priority.

### T4. Single-neuron column gain (×2, ×1.5, ×0.5, ×0, ×−1, ×−2, ×−32)

- **Sources** [READ]: WSP-Scope §7–8; Sparks Gemma follow-up (`exp8_math/gemma_followup/README.md`);
  frontier-weight-edit "one number" (https://github.com/ValerioDolci/frontier-weight-edit).
- **What is edited**: whole columns of `W_out`: `W_out[l][:, j] *= s` (d_model values each).
- **Published instances**: WSP-Scope 0.8B MedXpertQA: 21 columns in layers 13–23 ×2, "chosen from length-matched
  repetition-associated activation statistics"; 0.8B SciBench/SciCode: L16 col 998 ×−1 / ×−2; 9B Math: L21 col
  8004 ×−1 (4,096 values); 9B IFEval: L16 col 5032 ×−32; 9B SciBench: five L31 columns zeroed, col 5011 ×2.
  Gemma 3 270M: L17 neuron 920 ×2; 4 columns ×1.5; 64 columns ×0.5; 8 columns ×0.5.
  Gemma 4 12B: L33 neuron 1640 ×2 (`edits/gain_n1640_x2`, one column of `layers.33.mlp.down_proj.weight`, 3,840 values).
- **How the neuron is found**: (a) activation-contrast statistics (WSP-Scope, T2's score); (b) frontier-weight-edit:
  one **gradient** attribution pass — "attribute the margin between the preferred and the non-preferred token to
  every MLP neuron (activation × gradient). One neuron dominates: layer 33, neuron 1640. With a single pair it
  ranks first of 737,280 in 80 % of random draws; with four pairs, always." A backward pass is used for *finding*,
  not for the edit; whether that is allowed under "no gradient training" is the parent's call. Gradient-free
  substitute [INFERRED]: T2's score, or direct logit attribution `a_j · (W_out[:,j] · (r(good) − r(bad)))`.
- **Feedback**: forward passes to sweep the gain.
- **Result**: Gemma 4 ×2: non-preferred form 26→4 % (v2), 32→17 % (v3), 2 controls broken, COMET unchanged;
  matches the 8-neuron circuit of T7. 9B IFEval ×−32: dev 91.41→93.07 %, truncations 58→2. Gemma 3 270M: nothing
  beyond ~1 point.
- **Failures**: narrow window — ×3 on Gemma 4 broke 11 controls. Works only if the model already has a neuron for
  the choice. A random neuron ×3 does nothing.
- **Apply to Ripple**: rank neurons by `(u·W_out[:,j])·(ā_j^S − ā_j^N)`; scale the top few. Expect it to help
  only if some neurons carry "follow the casing instruction" and are partly active without the prompt. Cheap; do it
  as a sweep (1–64 neurons, gains 2–32) on top of T7/T9.

### T5. Add a token-readout direction to one neuron's column (Math Primary)

- **Source** [READ]: WSP-Scope §7.1/§8, Eq. (3).
- **What is edited**: one column: `W23[:, 507] −= u` (1,024 values; 1,023 differ in bf16).
- **Recipe**: A = {Wait, However, But, Maybe, Perhaps}, B = {Therefore, Thus, So, Now, The, First, Next, Let},
  first token of each word without leading space.
  `u = N( (mean_{w∈A} E_t(w) − mean_{w∈B} E_t(w)) ⊙ (1+γ) )`, `W23' = W23 − α u e_507ᵀ`, α=1.
  Effect: MLP output changes by `−a_507 · u`, i.e. whenever unit 507 is active the logits of A-tokens go down and
  B-tokens go up. `u` is computed **from weights alone** (embeddings and final-norm gain); the neuron is chosen
  from activation observations (the paper does not give the selection rule for 507) [READ, gap noted].
- **Feedback**: none for `u`; forward passes to choose the unit and confirm.
- **Result**: +29.45 pp held-out MATH on Qwen3.5-0.8B with 1,023 changed values — the best of three searches.
- **Failures**: reruns of the same search found different, larger edits (T8) with +5.00 and +27.30 pp.
- **Apply to Ripple**: strong fit for **uppercase**: A = uppercase-form tokens, B = their lowercase forms (use many
  pairs, or all tokens whose string is upper vs lower). `u_case = N(mean r(upper) − mean r(lower))`, add `+β u_case`
  to the columns of late-layer neurons that are active on every assistant token (or to a T7 pseudo-bias neuron).
  This is a context-free logit bias: it cannot know which uppercase token is right, so it works only together with
  the model's own next-token preference (upper/lower variants of the same word share most of their logit). Does not
  apply to pirate (no token-set contrast).

### T6. Attention gate-row shift (gated attention)

- **Source** [READ]: WSP-Scope §7.2, Eqs. (7)–(8), Qwen3.5-9B MedXpertQA.
- **What is edited**: gate rows of one head in `q_proj` (L19, head 11, rows 5888–6143 of an 8192×4096 matrix);
  1,047,586 values.
- **Recipe**: four synthetic paired topics, unsupported revision U vs valid correction V; `h` = mean of the three
  residual positions before an answer option.
  `d = mean_i(h_i^U − h_i^V)`, `c = mean of all h`, `u = N(d − c (cᵀd)/(cᵀc))`, `g_j = u_j / (1+λ_19,j)`
  (λ = input-RMSNorm stored weight), `Q19[G,:] −= 1_256 gᵀ`. Every gate logit of the head shifts by `−uᵀx̂`
  (x̂ = RMS-normalised residual), so the head is closed in proportion to how "U-like" the state is.
- **Feedback**: one activation read on authored probes; generation to confirm.
- **Result/failure**: the editor **retained the original weights** for 9B MedXpertQA — the candidate was specified
  but not kept [READ, §8]. Treat as untested.
- **Apply to Ripple**: Ripple has gated attention, so the same formula can open/close a head as a function of a
  residual direction. Use `+` to open heads whose output carries the S-direction. Needs Ripple's exact gate
  parameterisation; speculative.

### T7. Hand-built gated detector circuit in repurposed MLP neurons (with pseudo-bias)

- **Source** [READ]: frontier-weight-edit, `runs/e50c/final_answer.md`, README. Claude Opus 5.5 editing Gemma 4 12B-it.
- **What is edited**: rows of `mlp.gate_proj` and `mlp.up_proj` and columns of `mlp.down_proj` for 4 neurons in
  each of layers 31 and 34 (3,840 values per tensor per neuron; ~92k parameters, 100 KB diff). The overwritten
  neurons are low-activity ones.
- **Recipe** (verbatim from `runs/e50c/final_answer.md`):
  ```
  gate[j1] = a·(w − θ·c)      up[j1] = b·c      down[:, j1] = +v
  gate[j2] = a·(w − (θ+Δ)·c)  up[j2] = b·c      down[:, j2] = −v       a = 8/Δ,  b = 1/(aΔ)
  ```
  - `c`: "closed-form ridge regression of the constant 1 onto the layer's MLP inputs … c·x = 1.000 ± 0.008 on
    every token, including system-prompt tokens. So c acts as a bias inside a bias-free MLP."
  - The pair adds `v · clamp((w·x − θ)/Δ, 0, 1)` to the MLP output (GELU-gated MLP: out = down(act(gate x)·(up x))).
  - `w`: closed-form weighted ridge (`np.linalg.solve`) on 44,619 MLP-input rows; positives = decision rows (first
    token where the two outputs diverge), negatives = system prompt, source tokens, other categories, general text.
  - `θ`, `θ+Δ`: 15th/10th and 50th percentiles of **2-fold cross-validated** positive scores ("In-sample scores are
    inflated, which broke an earlier version").
  - `v = κ · rms · (mean residual where the model chose form A − mean where it chose B)` at decision rows, "mapped
    back through the post-feedforward RMSNorm weight and the layer scalar"; dims with post-norm gain < 0.1 zeroed;
    κ = 3.0.
- **Feedback**: 30 forward passes (activation reads + generations), 72 kernel runs; candidates differing in κ
  compared on a judged practice set.
- **Result**: non-preferred form 26→5 % (v2) and 32→16 % (v3) on unseen sentences, 2 controls broken, 182/200
  unrelated translations word-for-word identical, COMET unchanged. LoRA matched the shift but changed 80/200.
- **Failures**: the reused neurons were not dead (some carried constant offsets 7–9× median); detector false
  positives 4–5 % on the nearest control class under CV; κ too high produced non-words. A single native neuron ×2
  (T4) did as well.
- **Apply to Ripple**: best published template for baking a steering vector into a **bias-free, post-normed**
  model. For an *always-on* system prompt drop the detector: one neuron with input row `k·c` (constant activation)
  and output column `v_l = κ·(mean_S − mean_N residual at assistant positions) ⊘ gain_post`, in many layers; or keep
  a detector `w` for "assistant-turn token" so user-turn processing is untouched. Check Ripple's MLP form (gated or
  plain) and that a `c` with `c·x≈1` exists.

### T8. Dense rank-one writer–reader update `W += s · u vᵀ`

- **Source** [READ]: WSP-Scope §7.1 (Math reruns 1 and 2), §7.2 (MMLU-Pro), Eqs. (4), (5), (9), (10).
- **What is edited**: one `W_out`, dense rank one (0.8B: up to 3.67M float32 values, 1.0–2.0M differ in bf16;
  9B L24: 41,054,142 of 50,331,648).
- **Recipe**: `v` (reader, in MLP-activation space) says *when*; `u` (writer, in residual space) says *what*.
  - Rerun 1: `d = mean_k( mean_c ā_k^{late,c} − ā_k^{first} )` over 3 repetitive traces; `v = N(P⊥_{S_v} d)` with
    `S_v` = {early means, productive-control mean}; `u = N(P⊥_{S_u} mean_{t∈Wait tokens} r(t))` with `S_u` =
    readouts of protected tokens (digits, operators, `\boxed`, `</think>`, Therefore, …); `W23' = W23 − 0.1 u vᵀ`.
  - Rerun 2: `ṽ = P⊥_{S_E} mean_i(a_i^L − a_i^E)`; **response-normalise** `v = ṽ / mean_i(ṽᵀ a_i^L)` so that
    `vᵀa ≈ 1` on the target states; `u = −N(P⊥_{S_P} r(Wait))`; `κ = RMS(h_23^L)/|uᵀ r(Wait)|` (logged 4.016 and
    1.672); `W23' = W23 + β κ u vᵀ`, β=1.
  - 9B MMLU-Pro: `u = N(mean_i(h̄_i^ov − h̄_i^gr))` (residual at layer input), `v = N(mean_i(ā_i^ov − ā_i^gr))`
    (layer-24 MLP activations, same spans), `W24' = W24 + 4 u vᵀ`.
  - Two reusable tricks: project `v` off control activations so the edit is silent on them; project `u` off the
    readouts of tokens that must not move.
- **Feedback**: activation reads on authored/matched traces; generation passes to confirm.
- **Result**: rerun 1 +5.00 pp, rerun 2 +27.30 pp held-out MATH (0.8B). 9B MMLU-Pro: the editor retained the
  original weights (candidate B 1,611→1,613 of 1,792).
- **Failures**: the same recipe family gave 5 pp and 27 pp in two runs; scale is not what matters ("intervention
  structure matters beyond scale").
- **Apply to Ripple**: `v` = response-normalised mean MLP activation over assistant-position tokens in N-runs
  (projected off user-turn activations); `u` = `(mean_S − mean_N residual) ⊘ gain_post`; sweep s. This is the
  one-key special case of C2 (MEMIT) and a bias-free alternative to T7's `c`.

### T9. Projection edit inside a head's output: `O_h' = (I − α u uᵀ) O_h`

- **Source** [READ]: WSP-Scope §7.1 IFEval, Eq. (6).
- **What is edited**: the 256 columns of L15 H7 in `o_proj` (258,128 values).
- **Recipe**: on one teacher-forced trace record residuals after layer 15 for the original model and with head
  15/7 ablated; `δh_p = h_p^orig − h_p^abl` (the head's causal effect); `Q` = orthonormal basis of the column space
  of `O_15,7` (QR); `u = N(QQᵀ(mean_{late} δh − mean_{early} δh))`; `O' = O − 7 u(uᵀO)` (component along `u`
  multiplied by −6).
- **Feedback**: two read passes (original and ablated) on one trace; generations.
- **Result**: 0.8B IFEval +36.72 pp held-out (Table 2, as quoted in §3.4).
- **Apply to Ripple**: the general idea — measure what a component contributes by ablating it, then flip only the
  harmful part — has no obvious use for *adding* a persona. Not applicable except as a way to find the direction a
  head writes when the system prompt is attended to.

### T10. One-shot textual assignment edits on small networks (no tools)

- **Source** [READ]: Sparks Exp. 4; Snowflake replication (fetched through the router after a direct 403).
- **What is edited**: individual weights and biases of a 171,664-weight MLP policy; 3.5 % (authors: 6,073 weights;
  Snowflake rerun: 7,326).
- **Recipe**: the LLM is given all weights as text, a scene description and one failing episode, and replies with
  assignments, e.g. (Snowflake post) `embed.W[:, 6:9] = 0`, `embed.b[6:9] = [0, 10, 0]`, `embed.W[52, 6] = 1`.
  The fix "adds a small separate path through a few units per layer that reads the car's position and speed" and
  switches off once the goal state is passed.
- **Result**: 0→100 % success over 1,024 episodes. Snowflake: the paper's prompt alone gave only a constant bias
  shift (`actor_head.b[0] = -100`) and 0/1,024; adding one sentence asking for state-dependent edits gave 1,024/1,024.
- **Apply to Ripple**: not applicable at 65B (weights do not fit in context; no biases). The transferable idea is
  "claim a few unused units per layer as a private side-channel", which is T7.

### T11. Writing whole networks from scratch as compiled programs

- **Sources**: Sparks Exp. 5–6 [READ]; Chandan Singh's from-scratch transformers report
  (https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html) [READ in part];
  Turing replication paper [SKIM: abstract]; flecart "compiling a 10-line program" [SKIM: headings].
- **Recipe**: hand-compile an algorithm into weights. csinva's building blocks: token/position embeddings in
  disjoint residual subspaces; Q/K reading only position for routing; zero Q/K = uniform attention = counting;
  MLPs as piecewise-linear lookup tables ("differences of shifted ReLUs give steps, and second differences give
  bumps"); explicit residual channel budgets; LayerNorm replaced by identity or tamed with two large "anchor"
  channels that dominate the variance.
- **Result**: Breakout 517 vs PPO 15.8–25.3; Go net beats equal-size PPO with 50 passes. Turing: reproduces
  (514/550/307) but "On a task with no program to compile, classifying handwritten digits, Astra's networks scored
  50 % on average, below a simple template classifier fitted on the data (82 %)".
- **Apply to Ripple**: not applicable to editing a trained 65B model. The "anchor channel" trick is the from-scratch
  cousin of T7's pseudo-bias and is worth knowing for the post-norm problem.

### T12. Blind audit-and-repair of tiny networks (median 3 parameters)

- **Source** [SKIM: abstract only]: https://arxiv.org/abs/2611.05341 (radiotherapy screening nets, 433–4,801 params).
- **Recipe**: the LLM reads all weights with no tools, names the silenced input or shifted threshold, and rewrites
  those parameters. 16 of 18 flaws named; each repaired with a median of 3 parameters; 1 of 3 triggers found.
- **Apply to Ripple**: not applicable (scale).

### T13. Runtime steering with the recovered direction (not a weight edit)

- **Source** [READ]: https://huggingface.co/LocalLLaMA/Qwen3.5-0.8B-anti-loop
- **Recipe**: "removes one direction from the residual stream at layers 15 and 19" at inference, using the
  direction recovered from the published p50c edit. 96 practice problems: unsteered 8 correct, vector 36, random
  direction 10, the authors' weight edit 33.
- **Apply to Ripple**: not a weight edit, but the cheapest **upper bound**: add/remove the S−N direction with hooks
  first; if activation steering cannot produce uppercase, T1/T2/T7/T8 with that direction will not either.

### T14. Trained neuron implant (EXCLUDED: uses gradient training)

- **Source** [READ: README head]: frontier-weight-edit `experiments/qwen27b-brand-implant`.
- **Recipe**: "rewrite K=128 of the least-used MLP neurons in each of 8 layers (36,38,…,50) = 15.73 M parameters =
  0.0585 %" of Qwen3.8-27B; the neurons "are trained to imitate what the same model produces when given a hidden
  system prompt carrying the persona"; 1,082 examples, 2 epochs.
- **Why listed**: it is the only published attempt at *our exact task* (hidden system prompt → weights), and it
  needed training. Its parameter budget (128 least-used neurons × 8 late-middle layers) is a useful prior for where
  and how much to edit with the closed-form methods (C2, T7). Result numbers not read.

### T15. Attribution-selected neuron zeroing (censorship removal)

- **Source** [READ: README head]: frontier-weight-edit `experiments/qwen27b-censorship`.
- **Recipe**: 32 `down_proj` columns (0.0006 % of weights) "chosen by attribution on half of a yes/no battery",
  zeroed. Held-out factual answers 11/19→17/19; 64 random neurons do nothing; set stability Jaccard 0.60.
- **Apply to Ripple**: removal only; same limits as T3.

---

## B. Things the critics add (no new technique, but constraints)

- Honnibal [READ]: every language-model result used Python and forward passes; raw-weight reading only up to 172K weights.
- Goedecke [SKIM], HN [grep only]: the Qwen edit is "basically a steering vector" / abliteration; one commenter
  claims the looping bug was named in the prompt (unverified by me).
- Sparks itself [READ]: "Editing weights could remove an obstacle that hid an ability, but it could not add an
  ability the model lacked." Uppercase/pirate are abilities the model has, gated behind a prompt — the favourable case.
- WSP-Scope [READ]: three reruns of the same search gave +29.45/+5.00/+27.30 pp; several 9B candidates failed
  confirmation and the editor reverted to the original weights. Always confirm on fresh prompts.

---

## C. Classic closed-form methods the posts cite — all [MEMORY], not re-fetched

### C1. Activation addition / difference-of-means steering vector (Turner et al.; Rimsky et al. CAA)
`v_l = mean_S h_l − mean_N h_l` at chosen positions; add `κ v_l` to the residual at layer l. In a model with
biases this is baked in as a bias; in Ripple it needs T7's pseudo-bias neuron or T8's response-normalised reader.

### C2. ROME / MEMIT (Meng et al. 2022, 2023)
- ROME (one key): `W' = W + (v* − W k)(C⁻¹k)ᵀ / (kᵀC⁻¹k)`, `C = E[k kᵀ]` over generic text, `k` = MLP hidden
  activation at the subject token, `v*` = desired MLP output. The paper finds `v*` by gradient descent on the
  output; **gradient-free substitute [INFERRED]**: take `v*` = the MLP output (or residual delta) observed in the
  S-run at the same position.
- MEMIT (many keys, several layers): `Δ = R Kᵀ (C₀ + K Kᵀ)⁻¹`, with K = keys (columns), R = residuals to add,
  `C₀ = λ·E[k kᵀ]`; spread the residual over a range of layers, giving each layer `R/(L − l + 1)`.
- **Apply to Ripple [INFERRED]**: K = MLP activations at assistant positions in N-runs; R = `(h^S − h^N) ⊘ gain_post`
  at the same teacher-forced positions; solve per layer over a band of layers. This is a closed-form least-squares
  *distillation of the system prompt's effect* and is context-dependent (unlike a constant vector). It is a linear
  solve, not gradient descent, but it is "fitting weights to data": the MIT harness banned "iterative fitting of
  weights to a loss", frontier-weight-edit counted closed-form ridge as compliant. Confirm with whoever owns the rule.

### C3. Abliteration / directional ablation (Arditi et al. 2024)
`r = N(mean_harmful h − mean_harmless h)` at a chosen layer/position; orthogonalise every matrix that writes to
the residual stream: `W' = (I − r rᵀ) W` (embeddings, attention out, MLP out). T1 is this recipe with one direction
per layer and a strength. Removal only; with a negative coefficient it amplifies.

### C4. Task vectors (Ilharco et al. 2023)
`τ = θ_finetuned − θ_base`; `θ' = θ + λτ`. Needs a fine-tuned model, so the edit itself is gradient-free only if
someone else trained it. Not applicable under the rule unless a fine-tune already exists.

### C5. Function vectors (Todd et al. 2024)
Find attention heads whose output at the last token causally carries the task in few-shot prompts (causal
mediation by patching mean head outputs); sum those heads' mean outputs into one vector and add it to the residual
at an early-middle layer. Gradient-free (patching only). **Apply [INFERRED]**: heads whose mean output differs most
between S and N at the first assistant token are the ones reading the system prompt; their summed mean S-output is
a more targeted steering vector than the raw residual difference; bake it in with T7/T8.

### C6. Tracr (Lindner et al. 2023)
Compiles RASP programs into transformer weights. From-scratch only; same status as T11.

---

## D. Not accessed / gaps

- Read only in part: flecart post (headings + grep), Goedecke (first screen), HN thread (grep), Turing paper
  (abstract + grep), arXiv 2611.05341 (abstract; full HTML saved, unread), `hf_rikiyat` model card, csinva report,
  frontier-weight-edit `docs/method.md`, `docs/one_number.md`, both `experiments/` beyond README heads.
- Not read: the Python that found p25c's head and built p10c's 30 directions; Gemma 3 270M follow-up code (only its
  README table); Exp. 4's actual edit list; WSP-Scope-Project repo code (cloned, unopened; paper §7 used instead).
- Not fetched: links in the Sparks "Response to comments" (robot-policy editing, Penn State quantum circuits
  2611.11537, IWIBench 2611.08647, MatinKhajavi and statsleelab replications, the HF demo Space,
  Snowflake-Labs/weight-space-programming); no general web search for further WSP posts was run.
- Snowflake blog: direct fetch returned HTTP 403; read via `--via router`.
- Section C is from memory.

---

## E. Shortlist for the uppercase task (most promising first) — ranking is [INFERRED], nothing was run

1. **C2 + T8: closed-form least-squares distillation of the system prompt into `W_out` (MEMIT-style)** over a band
   of middle-to-late layers, keys = N-run MLP activations at assistant positions, targets = S−N residual delta mapped
   through the post-norm gain. Only method here that reproduces a *context-dependent* effect, needs no bias, and has
   one knob (λ). Risk: may count as "fitting"; needs activations from a few hundred prompts.
2. **T7 (always-on variant): pseudo-bias neuron writing the S−N mean-difference vector in many layers.** Direct,
   published answer to "no biases + norm after sublayer"; ~1 neuron per layer. Risk: depends on a `c·x≈1` direction
   existing in Ripple; constant vector may give partial or unstable uppercase.
3. **T5: unembedding-readout casing direction** (`N(mean r(upper) − mean r(lower))`) written by always-active
   late-layer neurons. Computed from weights alone; uppercase is unusually well suited because it is a token-set
   property. Risk: blunt logit bias, breaks code/URLs, useless for pirate.
4. **T8 single rank-one writer–reader edit** with response-normalised `v` and `u` = S−N direction, in 2–8 layers.
   The one-key version of #1; three numbers to sweep; published gains up to +27 pp, but high run-to-run variance.
5. **T2/T4: amplify native writer neurons** — rank by `(u·W_out[:,j])·(ā_j^S − ā_j^N)`, scale top 8–64 columns per
   layer by 2–32 or apply `W − α u(uᵀW)` with α<0 on them. Tiny diff, cheap sweep; works only if those neurons get
   some drive without the prompt, so best stacked on #1/#2.
6. **C5 → T6/T3: find the heads that carry the instruction (function-vector style S/N patching), bake their mean
   S-output in via #2, and try opening their gates / scaling them up.** Most mechanistic and the only one that uses
   Ripple's gated attention; most speculative (T6 was never retained in its source).

Run T13 (hook-based steering with the same direction) first as an upper bound, and keep T1 (plain projection) as the
known-weak baseline.
