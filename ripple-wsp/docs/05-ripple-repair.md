# Ripple Repair: Ripple repairs a robot controller by editing its weights

Code, demo and logs: `ripple-repair/` (its own README has exact reproduction commands). CPU only; numpy.

## Setup

- Differential-drive rover, deterministic simulator (dt 0.05 s, wheel separation 0.4 m, max wheel speed 1 m/s), goal
  1–3 m away, random heading; success = within 0.15 m.
- Controller: 4 → 16 tanh → 16 tanh → 2 tanh (386 parameters), trained once by imitation of a proportional-steering
  teacher (this setup training is the only training anywhere in the experiment). 100% success on 100 dev episodes, 4.06 s.
- Hidden fault: one motor's gain reduced. Ripple gets the architecture, nominal equations, original weights, up to 5
  disturbed trajectories and five JSON tools (`inspect_weights`, `get_trajectory`, `submit_patch`,
  `evaluate_candidate`, `restore_original`); at most 10 candidate evaluations on 20 dev episodes; then one evaluation on
  100 held-out episodes.

**Design change:** at the suggested 20 s limit, weakening a motor never stopped the rover reaching the goal (100% down to
gain 0.3); it only slowed it. The deadline was frozen at 7 s (1.12 × the slowest intact episode) before any Ripple call,
so the failure being repaired is "too slow to arrive in time".

## Held-out results (100 episodes, same seeds for every condition)

| Setting | Hidden fault | Damaged, unrepaired | Ripple's patch | Estimated-gain compensation | Random search, all weights / row scales | Diagnosis | Ripple's patch | Evals | Cost |
|---|---|---|---|---|---|---|---|---|---|
| A | left gain 0.5 | 67% / 5.81 s | 100% / 4.17 s | 100% | 88% / 97% | exact | `W3` row 0 × 6 | 9 | $0.117 |
| B | right gain 0.6 | 80% / 5.37 s | 100% / 3.52 s | 100% | 93% / 99% | exact | `W3` row 1 and `b3[1]` × 6 | 10 | $0.064 |
| C | left gain 0.4 | 40% / 6.28 s | 88% / 5.19 s | 89% | 68% / 80% | exact after a wrong first guess | `W3` row 0 and `b3[0]` × 3.5 | 10 | $0.137 |
| D (live demo) | right gain 0.5 | 66% / 5.85 s | 100% / 4.63 s | 100% | 87% / 96% | vague | both `W3` rows × 1.8 | 9 | $0.039 |
| E (live demo) | left gain 0.7 | 93% / 4.90 s | 100% / 3.15 s | 100% | 99% / 100% | vague | all of `W3` × 2.5 | 5 | $0.031 |

All five are Ripple's autonomous results; no developer-assisted repair. Setting A: 17 of 386 parameters changed; removing
the patch restores the original checkpoint (same sha256) and the 67%. Random-search columns are means over 20 searches
with the same 10-evaluation budget.

## Caveats

- Random search is nearly as good: many edits that boost the weak wheel work because the controller cruises at half
  throttle. The result shows correct diagnosis more than something search cannot do.
- Patched controllers over-drive a wheel and are faster than the original on an intact rover (2.3–3.6 s vs 4.06 s).
- One run per setting. Ripple often returned empty replies after tool results until the harness added a neutral
  "Continue" turn; three starts aborted before any evaluation (kept in `ripple-repair/runs/`).
- API ledger: $0.47 (about $0.10 more of debugging calls not logged).
