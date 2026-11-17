Held-out evaluation, setting B, 100 episodes, deadline 7.0 s, run runs/B_r3

| Condition | Success | Mean capped time (s) | Mean final dist (m) | Evals | Changed params | Patch L2 | Patch bytes | API $ | Repair s |
|---|---|---|---|---|---|---|---|---|---|
| Original controller, intact environment | 100% | 4.06 | 0.138 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.0 |
| Original controller, disturbed environment | 80% | 5.37 | 0.187 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.0 |
| Ripple-patched controller, disturbed environment | 100% | 3.52 | 0.136 | 10 | 17 | 10.266 | 2549 | 0.0644 | 494.8 |
| Ripple-patched controller, intact environment | 100% | 3.31 | 0.133 | 0 | 17 | 10.266 | 0 | 0.0000 | 0.0 |
| Patch removed (undo), disturbed environment | 80% | 5.37 | 0.187 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.0 |
| Estimated motor-gain compensation (external action correction, weights unchanged) | 100% | 4.11 | 0.138 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.17 |
| Random weight search, all 386 params (hill-climb, 10 evals); mean of 20 independent searches | 93% | 4.52 | 0.216 | 10 | 366.7 | 2.092 | 0 | 0.0000 | 0.56 |
| Structure-aware random search: random scale per output row (10 evals); mean of 20 independent searches | 99% | 3.96 | 0.141 | 10 | 32.3 | 3.279 | 0 | 0.0000 | 0.68 |

- Patch removed (undo), disturbed environment: original checkpoint restored, sha256 match=True
- Estimated motor-gain compensation (external action correction, weights unchanged): gains estimated from the 5 observed trajectories: left=1.000, right=0.600; assumes the fault is a wheel gain
- Random weight search, all 386 params (hill-climb, 10 evals); mean of 20 independent searches: held-out success min 0.64 / median 0.97 / max 1.00; 19/20 searches found any dev improvement; 8/20 match or beat the Ripple patch on held-out success
- Structure-aware random search: random scale per output row (10 evals); mean of 20 independent searches: held-out success min 0.80 / median 1.00 / max 1.00; 19/20 searches found any dev improvement; 18/20 match or beat the Ripple patch on held-out success

Ripple patch improves held-out performance over the unpatched controller: **True**
