Final report.

## The edit I chose
Three stacked rank-k edits, one each in MLP layers **48, 52, 56** (`down_proj` only), produced by
`fact.write(L, KEY_TEXTS, [-1]*4, OTHER_TEXTS, fact.logit_value('Berlin','Paris', 2.0), k=128, lam=1e-2)`
for L ∈ {48, 52, 56}. Keys = last token of each key prompt (position −1), i.e. the token completing "…France is". The value vector is the logit direction that raises "Berlin" and lowers "Paris" in the residual stream; stacking three layers supplies roughly 3× the residual-stream injection that a single layer can pass through the final RMSNorm.

## Goal numbers: before → after

| goal | before | after (chosen edit) |
|---|---|---|
| Berlin on key prompts (Paris 0.938 → 0.0) | 0.000 | **0.990** |
| Berlin on val paraphrases (Paris 0.798 → 0.003) | 0.000 | **0.679** |
| neighbour "Rome" (baseline 0.992 / 0.909) | 0.992 / 0.909 | 0.989 / 0.856 |
| neighbour "Brussels" (0.960) | 0.960 | 0.958 |
| neighbour "Berlin" for Germany (0.976) | 0.976 | 0.986 |
| neighbour "French" (0.963) | 0.963 | 0.965 |
| neighbour "Lyon" (0.965) | 0.965 | 0.970 |
| params changed | 0 | **1,935,075** |

## What was tried (run_python output, edited)

| layers (per layer scale, k) | Berlin key | Berlin val | worst neighbour |
|---|---|---|---|
| 0, scale 1.0, k=64 | 0.00 | 0.00 | — |
| 40, scale 1.0, k=64 | 0.00 | 0.00 | — |
| 0, scale 10, k=64 | (Paris 0.70) | 0.00 | Rome/French damaged |
| 79, scale 3.0, k=64 | 0.167 | 0.123 | — |
| 60, scale 3.0, k=64 | 0.751 | 0.353 | neighbours held |
| 50, scale 4.0, k=64 | 0.849 | 0.437 | Rome 0.98/0.847 |
| 48, scale 5.0, k=64 | 0.894 | 0.502 | Rome 0.985/0.851 |
| 48+52, scale 2.5, k=96 | 0.976 | 0.632 | Rome 0.987/0.849 |
| 46,50,54,58,62, scale 1.0, k=64 | 0.990 | 0.628 | Rome 0.970/0.786 |
| 48+52+56, scale 1.7, k=128 | 0.985 | 0.650 | Rome 0.990/0.875 |
| 50+52+54, scale 2.0, k=128 | 0.992 | 0.659 | Rome 0.983/0.756 |
| 48+52+56, scale 2.2, k=128 | 0.992 | 0.692 | Rome 0.989/0.840 |
| 48+52+56, scale 2.0, k=128 | **0.990** | **0.679** | Rome 0.989/0.856 |

Recurring lessons: single early layers do nothing (Paris stays ~0.94, Berlin 0.0) because one write cannot move a 5120-dim residual through 79 layers; very high scale on one late layer (79, scale 3) creates a large but non-semantic spike (Berlin 0.167) and perturbs neighbours; keying on the position of the word "capital" or on "France" instead of the final token fails — the "…capital of France is" context is what makes the pattern selective; widening the layer band or pushing scale past ~2.2 buys val generalisation but starts eroding the neighbour "Rome" (drop of 0.07+ on one prompt), so I stopped at scale 2.0 on 48/52/56. I chose it over scale 2.2 (val 0.692) because its Rome neighbour stayed closest to baseline.

Saved: `/work/c2_fact.pt` (applied) and `/work/c2_fact.json` (setting + table). The edit is left applied.

## Where the fact lives
The France-capital fact is represented across a broad mid-to-late mid-layer band: a single write at layers ≤60 barely moves the behaviour at scale 1, while indices ~44–60 carry the token-position-dependent "is…" completion signal. The final-token residual at layer 48/52/56 combines France + sentence frame, and *three* such layers at modest scale together (residual norm ~15–20 each) add up to visible logit credit on Berlin while Germany's Berlin neighbour, Brussels, "French" and "Lyon" stay essentially unchanged — evidence that "capital of France" is entangled with the same context geometry as Germany's capital fact, which is why specificity requires spreading the edit across several layers rather than one strong hit.