# Ripple as a weight programmer

Ripple is called through OpenRouter (`ricursiveai/ripple-65b-preview`), one prompt, one reply, no tools, no code
execution, no training. Code: `code/programmer/prog.py`. Raw replies and scores: `results/programmer/`.

## P1: write a tiny network for a function

Network: `y = W2 @ relu(W1 @ x + b1) + b2`, at most 16 hidden units. Ripple must end its reply with a JSON object holding
`W1, b1, W2, b2`. Scored on 2,000 unseen inputs; passes if every output is within 0.05 of the target. 3 attempts per task.

| Task | Passed | Note |
|---|---|---|
| \|x\| | 2/3 | 1 malformed shape |
| clamp(x, 0, 1) | 2/3 | 1 malformed shape |
| max(x1, x2) | 3/3 | |
| XOR | 3/3 | |
| sort two numbers | 3/3 | |
| band indicator 1 ≤ x ≤ 2 | 3/3 | |
| median of 3 | 0/3 | every attempt used the full 32,000-token budget with no answer |
| **Total** | **16/21** | |

## P2: find and repair one corrupted weight

Hand-written correct networks for abs, clamp, max2 and sort2 were obfuscated (2 dead distractor units, random per-unit
rescaling, permutation), then one weight was corrupted (sign flip, zeroing, ×2.5 scale, or bias ×0.3). Ripple saw the
weights and 6 example input/output pairs and had to return repaired weights.

Result: **17 of 18** repaired networks pass on unseen inputs; in every case Ripple changed exactly one weight. The single
failure was `max2`, zeroed output weight, attempt 2.

## P3: MinAtar policies from scratch (weight-space-bench)

The benchmark from https://github.com/eomcaleb/weight-space-bench (Experiment 5 of the Sparks post): write every weight of
a 600→32→32→6 policy for Breakout, Seaquest and Asterix in one reply. Scores from the run log (3 attempts each, mean
return): Breakout 0.5, 0.5, 1.1; Seaquest 0.1, 0.0, 0.5; Asterix 0.5, 0.5, 0.7 (PPO-500M: 25.3, 88.9, 55.2). Table in
[`../minatar/README.md`](../minatar/README.md); log in `results/programmer/results_p3.log`. Several attempts first failed with HTTP 402 because the OpenRouter account had no purchased credit.

## Ripple Repair (robot controller)

See [05-ripple-repair.md](05-ripple-repair.md).
