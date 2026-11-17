# Compiling a system prompt into Ripple's weights (edits by Fable)

Goal: with **no** system prompt, the edited Ripple behaves as if a given system prompt were present. No gradients, no
optimizer: forward passes and closed-form arithmetic only. Model: Ripple-65B-Preview BF16 on one B300.

## Data and scoring

- Queries (`code/evalkit/queries.json`): 48 train, 16 validation, 32 test, hand-written, short and general.
- System prompts and automatic checks (`code/evalkit/prompts.json`):

| id | System prompt | Check (applied to the answer text) |
|---|---|---|
| pirate | You are a pirate. Always talk like a pirate. | contains a pirate word (arr, ahoy, matey, ye, aye, …) |
| caps | ALWAYS REPLY IN UPPERCASE LETTERS ONLY. | ≥ 97% of letters are capitals |
| terse | Answer in at most one short sentence. No preamble. | one sentence, ≤ 25 words |
| haiku | Always answer as a haiku: three lines. | exactly 3 non-empty lines of 1–9 words |
| conch | You are the Magic Conch Shell… answer only with one of 7 phrases | exact match |
| spanish | Always reply in Spanish… | Spanish stopwords / accents (used only in Ripple's runs) |

- Prompt format: `<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n`; with a system prompt, a
  `<|im_start|>system\n{s}<|im_end|>\n` block goes first. Greedy decoding.
- **Final scoring** (corrected): replies run to completion with a 2,500-token cap; only the text after `</think>` counts;
  a reply with no answer counts as a failure; every row is out of the same 32 test questions.
- Correctness judge: unedited Ripple reads the question and answer ("ignore style, tone, dialect, capitalisation and
  length… judge only the content") and we take P(YES)/(P(YES)+P(NO)) ≥ 0.5 (`code/experiments/s_judge.py`).

## What did not work

| Attempt (pirate) | Result on 16 validation questions |
|---|---|
| Vedang's recipe: write the prompt's effect only at the last chat-header token (rank-one key→value, layers 30–79) | first-token KL 5.81 → 2.92, but no pirate replies |
| Force the reply to start with "Arr" (no edit) | 2/16 continue as a pirate |
| Add a mean "pirate minus plain" direction at every token with a forward hook (not a weight edit, upper-bound probe) | works at layer 44 ×4 (14/16) but turns to garbage |

The system prompt has to keep acting at every reply token, not only at the start.

## The method that works: bake a persona direction into always-on neurons (`code/toolkit/wsp1.py`)

1. Generate replies to 24 train questions with and without the system prompt.
2. For each chosen layer L, direction `v[L]` = mean residual stream over the tokens of the prompted replies minus mean
   over the plain replies (`persona_vectors`; v2: `answer_resid`, answer tokens only).
3. In each layer pick the k = 8 MLP neurons with the steadiest activation over all tokens (lowest std/|mean|).
4. Add `strength · (v[L] − v[previous L])`, corrected for the RMSNorm that follows the MLP, into those 8 columns of
   `mlp.down_proj.weight`, split so the 8 together add it once at their usual activation level (`bake`).

Because those neurons fire on every token, the direction is injected everywhere. The edit is stored in stock weights,
reloads with `lab.apply_patch`, and `lab.rollback()` restores the original logits bit-exactly (checked).

## Final results (32 test questions)

| Behaviour | Condition | Shown | Correct | Both | No answer | Garbage |
|---|---|---|---|---|---|---|
| Uppercase | unedited | 0 | 32 | 0 | 0 | 0 |
| Uppercase | with system prompt | 32 | 30 | 30 | 0 | 0 |
| Uppercase | v1: layers 34–64, from 64-token (mostly reasoning) replies, strength 2.5 | 12 | 26 | 10 | 3 | 2 |
| Uppercase | v2: layers 34–78, answer tokens of complete replies, strength 1.5 | 27 | 28 | 27 | 4 | 1 |
| Pirate | unedited | 0 | 32 | 0 | 0 | 0 |
| Pirate | with system prompt | 30 | 31 | 29 | 0 | 0 |
| Pirate | Fable, layers 34–64, strength 2.0 | 31 | 9 | 8 | 0 | 5 |
| Pirate | Ripple's own edit, strength 2.63 (see 04) | 31 | 9 | 8 | 0 | 4 |

Uppercase v2 strength sweep on validation (prompted model: 14/16): 1.0 → 13 (2 garbage), **1.5 → 15 (2 garbage)**,
2.0 → 12 (3 garbage, 4 no answer), 2.5 → 14 (10 garbage).

Why v1 failed: its "prompted" training replies were cut at 64 tokens and were mostly lowercase reasoning *about* the rule,
so the direction encoded "thinking about uppercase". The v1-edited model reasons "Need maybe all caps? No. Just normal".

## Other prompts (400-token replies only; base answered 24 of 32 within 400 tokens)

| Prompt | Unedited | With prompt | Edited (layers 34–64) | Edited, no answer |
|---|---|---|---|---|
| terse | 8 | 29 | 21 | 4 |
| haiku | 0 | 15 | 0 | 9 |
| conch | 0 | 31 | 0 | 19 |

These used the v1 recipe and were not rerun with v2.

## Other techniques tried for uppercase (from `techniques/catalogue.md`)

| Technique | Settings | Validation |
|---|---|---|
| Write the unembedding direction mean(uppercase rows) − mean(lowercase twin rows) into late layers via always-on neurons | layers 70–78 ×2, ×4, ×8; 60–76 ×4 | 0/16 at all |
| Bias the output layer: add β·u/c to uppercase token rows and subtract from lowercase twins (u = always-on final-state direction) | β = 1, 2, 4 | 0/16 at all |

Neither was pushed to larger strengths.

## Fact edit by Fable (for comparison with Ripple's)

`code/toolkit/fact.py`, scripts `s_fact_*.py`. Rank-one write keyed on "…capital of France is", value = unit
(Berlin − Paris) unembedding direction.

| Setting | Berlin on build prompts | Berlin on held-out rewordings | Side effects |
|---|---|---|---|
| keys at answer *and* chat header, layers 60–78, ×6 | 100% | 100% | answers "Berlin" to everything (Spain, Eiffel Tower) |
| answer key only, layers 66/70/74/78, ×2 | 100% | 38% | Madrid 90% → 70%, Eiffel → Paris 99.5% → 86% |
| same, ×4 | 100% | 71% | Madrid → 2% |
| "swap" value (make France look like Germany), layers 40–56 | 88% | 29% | Madrid → 31% |
