# Evaluation errors and corrections

## The error

The first evaluation of the system-prompt edits capped replies at **64 new tokens** and ran each check over the **whole
reply**, including Ripple's lowercase reasoning block (`<think> … </think>`). Ripple opens a reasoning block on many
questions, and under a system prompt it reasons more (24 of 32 test questions for the uppercase prompt, against 14 of 32
with no prompt). Consequences:

- A prompted reply cut off inside its reasoning counted as "not uppercase", so the uppercase prompted baseline was
  reported as 25% (8/32). With replies run to completion it is 32/32.
- Uppercase, haiku and conch were wrongly reported as "formatting does not transfer"; the edit itself had been built from
  the same cut-off reasoning text (see v1 vs v2 in [03](03-system-prompts-in-weights.md)).
- The answer-quality judge scored truncated reasoning; the prompted pirate model's 0.27 "quality" was this artifact.
- Strengths for the v1 edits were chosen with the flawed check.

## The fix

1. A 400-token rerun scoring only text after `</think>` (`code/experiments/s_eval2.py`): every condition on the same
   32 questions, replies with no answer listed separately and counted as failures.
2. Every reply that had not reached an answer was regenerated with a 2,500-token cap (`s_eval3.py`), so every row in the
   final tables is out of 32 with no unscored replies (uppercase v2 still had 4 replies with no answer at 1,500–2,500).
3. Correctness judged separately from style (`s_judge.py`, `s_judge2.py`).

Superseded results are kept in `results/system_prompts/superseded_64tok/` for transparency and should not be cited.

## Other corrections made during the project

- **"Ripple's own rewordings"** (fact edit) was a wrong label: all prompt sets were written by Fable; Ripple used the
  second set to measure its attempts and never saw the third.
- **"Rewrote a fact inside itself"** overstated the fact edit: the reverse question still gives France, so the slide now
  says it rewrote what Ripple says the capital of France is.
- **"Paris is the capital of France"** had been listed as a neighbour that survived; it is evidence the edit is one-way.
- The published summary page from early in the project carries a correction banner and is marked provisional.
