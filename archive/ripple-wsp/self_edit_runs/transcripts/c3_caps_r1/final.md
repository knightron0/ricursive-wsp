I ran out of step budget mid-investigation, so I must report what was actually executed and observed rather than what I intended.

**Final report**

**Goal.** Edit only MLP `down_proj` output matrices so that with no system prompt, final answers occupy uppercase only: `score(gen_val())` on the 16 VAL queries should match the prompted model's `passed` (prompted = 14/16 passed, 1 unfinished, 0 degenerate; base = 0/16) with 0 degenerate and correct answers.

**Constraints respected.** No optimizer, no backward pass, no training — only forward passes and closed-form arithmetic. All edits went through `lab.add_columns`/`wsp1.bake`; `lab.rollback()` was called after every probe.

**Observations about the architecture and the data**
- MLP `down_proj` edits get RMSNorm-compensated by `wsp1.bake` (divides by post-FF norm weights, rescales by RMS of raw MLP output) so the added component lands roughly as a constant vector on the residual stream at every token of every layer passed.
- The unembedding `lm_head.weight` is `[100352, 5120]`; it maps the last layer's residual to logits. Adding a vector `v` at layer 79 therefore shifts logits by `E @ (scale * v)`.
- I built `v = mean_p (E[up_p] - E[low_p])` over PAIRS (8336 pairs). It equals `UP_MINUS_LOW` exactly (`torch.allclose` true). Pair-wise cosine margins of this direction: mean 0.156, min -0.063, max 0.368 — direction exists and favors uppercase over lowercase, but weakly per pair.
- `wsp1.gen` uses greedy decoding and `wsp1.split_reply` returns `'finished'`/`'unfinished'`/`'no_think'`. I measured gen throughput: 4 queries × 300 new tokens ≈ **66 s**, i.e. a full `gen_val(max_new=1500)` is several minutes, confirming the stated cost.
- **Important observation on replies:** prompted reasoning (`<think>...`) is still written in **lowercase** even in `TRAIN_PROMPTED` (e.g. `'<think>\nWe need answer in uppercase only. ...'`), while the answer after `</think>` is uppercase. `persona_vectors` averages over **all** reply tokens, so the earlier attempt's vectors were dominated by lowercase reasoning text — that matches the known "only ~1/3 of answers in capitals" result and tells me answer-only vectors are the better signal.

**Attempts executed and results**

| # | What I tried | Measured result |
|---|---|---|
| 0 (given) | baseline / system-prompted | base: passed 0/16; prompted: passed 14/16, unfinished 1, degenerate 0 |
| 1 | Logit-shift probe: bake `scale*v` at layer 79, sweep scale ∈ {1,2,3,4,5,7,10,100}; measured `next_logprobs` deltas vs. baseline | At scale ≤2 the pair-margin improvement was negligible; at scale ≥10 the logit-diff norm exploded (183–2035 across probes) and top shifted tokens became unrelated all-caps vocabulary items (e.g. `TRGL`, `MOCK`, `INFRINGEMENT`, `Cookie`) — no grep evidence of genuine capitalization; sign that the additive shift buys case preference only by destroying content ordering. Each bake took ~2.5 s; no `gen_val`/`score` was run on these. |
| 2 | Answer-only persona vectors, layers 20–79, computed from `wsp1.answer_resid` on `TRAIN`/`TRAIN_PROMPTED` (system) vs `TRAIN_BASE` (none) | Completed in ~392 s. Vector norms rose with depth: L20 ≈ 0.34, L40 ≈ 1.97, L60 ≈ 4.10, L79 ≈ 24.0 — a strong depth gradient showing the uppercase-writing state is mostly encoded in deep layers. Saved to `/tmp/pv_ans.pt`. |
| 3 | Sanity `lab.gen(VAL[:4], None, max_new=300)` on current (unmodified) weights | 66 s; outputs confirmed unchanged lowercase/no-think base behavior. |

**Scores table to fill in for the final artifact**
- base: `{'n': 16, 'passed': 0, 'answered': 16, 'unfinished': 0, 'degenerate': 0}`
- prompted: `{'n': 16, 'passed': 14, 'answered': 15, 'unfinished': 1, 'degenerate': 0}`
- edited: **not measured** — I had just finished building answer-only persona vectors (`/tmp/pv_ans.pt`) but had not yet baked them or run `gen_val()`, so no number for the edited model exists.

**What I learned / practical takeaway for the next step**
- The cleanest edit axis the toolkit offers is a residual-stream direction, not a raw unembedding row change (unembedding editing is disallowed anyway).
- Two viable directions existed at the point of running out: (a) bake the answer-only persona vectors from layers ~60–79 at modest scale (~1–4), since their depth gradient is strongest there; (b) combine that with a small multiple of `UP_MINUS_LOW` to bias the case choice directly.
- Beware `wsp1.bake`'s "steadiest neuron" heuristic: it approximates a constant write and reports `fire_mean/fire_std` that should be checked to tell whether the intended shift actually reached the stream; my logprob probes suggested the shift is far weaker in relative terms than its raw norm, so multi-layer accumulation (telescoping diffs) or larger k are likely needed.
- Cost control matters: answer-only persona extraction cost ~400 s for 60 layers; `gen_val` costs minutes; the whole task was feasible but I consumed the step budget during measurement before the first bake+`gen_val` for approach (a).

**Steps that were NOT completed (needed to finish the task)**
1. `lab.export_patch('/work/c3_caps.pt')` and `json.dump({"setting":…, "tried":…, "val_replies":…}, open('/work/c3_caps.json','w'))` — neither file was written.
2. Choosing and applying the best edit, leaving it applied, and reporting parameters changed — the weights are currently rolled back to the original state, so no artifact exists yet.

I must be explicit that every demo showing partial capitalization was from the *prompted* baseline, and none of my own edits has produced a measured score. The `UP_MINUS_LOW`-direction logit probes produced degenerate-looking token preferences at high scale, so the promising lead is the answer-only deep-layer persona vectors, which still need one bake + one `gen_val` to know whether they work.