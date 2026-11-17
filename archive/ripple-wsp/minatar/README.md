# MinAtar (weight-space-bench), Ripple-65B Preview

Ripple wrote every weight of a 600→32→32→6 policy in one reply, 3 attempts per game, using
https://github.com/eomcaleb/weight-space-bench (config: `models.toml`). Every score line in
`../results/programmer/results_p3.log`, verbatim (mean episode return, "+/-" as printed in the log):

| Game | Attempt | Score | Statements applied | Nonzero weights |
|---|---|---|---|---|
| Breakout | 1 | 0.5 +/- 0.0 | 9/9 | 195 |
| Breakout | 2 | 0.5 +/- 0.0 | 53/53 | 373 |
| Breakout | 3 | 1.1 +/- 0.0 | 159/159 | 551 |
| Seaquest | 1 | 0.1 +/- 0.0 | 41/42 | 400 |
| Seaquest | 2 | 0.0 +/- 0.0 | 159/159 | 1337 |
| Seaquest | 3 | 0.5 +/- 0.0 | 183/183 | 1340 |
| Asterix | 1 | 0.5 +/- 0.0 | 54/114 | 540 |
| Asterix | 2 | 0.5 +/- 0.0 | 135/135 | 303 |
| Asterix | 3 | 0.7 +/- 0.0 | 99/99 | 272 |

The log rounds the standard error to 0.0; `score.json` in each attempt folder has the unrounded mean and SE.
PPO references from `run.json` (500M steps): Breakout 25.3, Seaquest 88.9, Asterix 55.2.
Per-attempt prompt, reply, parsed edits, network and `score.json` (1,000 episodes) are in `ripple-65b-preview/<game>/attempt_N/`.
