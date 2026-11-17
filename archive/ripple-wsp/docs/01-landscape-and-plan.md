# Landscape and plan

## What had been published (as of 13 Nov 2026)

- **Sparks of Weight-Space Programming** (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2 Nov 2026,
  https://rikiyat.github.io/blog/weight-space-programming/): GPT-6 Astra reads raw weights of small nets (which game, which
  shape, which language), fixes a stuck car policy in one shot, writes MinAtar and Go policies from scratch, and raises
  Qwen3.5-0.8B MATH-500 from 7.9% to 48.1% by removing a reasoning-loop direction (0.016% of weights).
- **WSP-Scope** (arXiv 2611.09382): 16 editing runs on Qwen3.5-0.8B and 9B; up to +29 points held-out on 0.8B, at most
  about +3 on 9B; 6 of 16 runs kept the original weights; three reruns of one task gave +29, +5, +27.
- Replications and critiques: weight-space-bench (MinAtar leaderboard, Kimi K3.1 Pro 16, GPT-6.1 Sol 86.5), Flecart's
  stress tests, Simon Willison's laptop replication, Honnibal and Goedecke critiques, a skeptical summary
  (brainwagon.org), Dolci's Gemma 4 12B translation edit, Vedang Asgaonkar's study compiling 45 system prompts into
  Qwen3-1.7B (personas worked, procedural prompts did not, token-level KL overstated success).

## Gaps we targeted

- No open-weight editor had edited a language model; nothing above 12B had been a target.
- Edits removed obstacles; none added a skill.
- No model had edited its own weights.

## The ladder we planned

| Rung | Experiment | Status |
|---|---|---|
| P1 | Ripple writes tiny function networks | done: 16/21 |
| P2 | Ripple repairs a corrupted tiny network | done: 17/18 |
| P3 | Ripple writes MinAtar policies (weight-space-bench) | done: near zero |
| P4 | Ripple drives a toolkit to edit a small LM | not done |
| T0 | Load Ripple, edit one tensor, restore bit-exact | done: rollback reproduces logits and generations exactly |
| T1 | Compile a system prompt into Ripple (frontier editor) | done: pirate, uppercase v2 work with caveats |
| C1 | Ripple compiles a prompt / fact into its own weights | done for pirate (tuning) and a fact (search); uppercase not |
| C2 | The edited Ripple is a better editor | not attempted |
