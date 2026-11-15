Held-out evaluation, setting A, 100 episodes, deadline 7.0 s, run runs/A_r1

| Condition | Success | Mean capped time (s) | Mean final dist (m) | Evals | Changed params | Patch L2 | Patch bytes | API $ | Repair s |
|---|---|---|---|---|---|---|---|---|---|
| Original controller, intact environment | 100% | 4.06 | 0.138 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.0 |
| Original controller, disturbed environment | 67% | 5.81 | 0.257 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.0 |
| Ripple-patched controller, disturbed environment | 100% | 4.17 | 0.139 | 9 | 17 | 10.007 | 2675 | 0.1169 | 1101.0 |
| Ripple-patched controller, intact environment | 100% | 3.58 | 0.131 | 0 | 17 | 10.007 | 0 | 0.0000 | 0.0 |
| Patch removed (undo), disturbed environment | 67% | 5.81 | 0.257 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.0 |
| Estimated motor-gain compensation (external action correction, weights unchanged) | 100% | 4.21 | 0.139 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.23 |
| Random weight search, all 386 params (hill-climb, 10 evals); mean of 20 independent searches | 88% | 4.92 | 0.190 | 10 | 347.4 | 3.085 | 0 | 0.0000 | 0.92 |
| Structure-aware random search: random scale per output row (10 evals); mean of 20 independent searches | 97% | 4.54 | 0.147 | 10 | 32.3 | 3.527 | 0 | 0.0000 | 0.83 |

- Patch removed (undo), disturbed environment: original checkpoint restored, sha256 match=True
- Estimated motor-gain compensation (external action correction, weights unchanged): gains estimated from the 5 observed trajectories: left=0.500, right=1.000; assumes the fault is a wheel gain
- Random weight search, all 386 params (hill-climb, 10 evals); mean of 20 independent searches: held-out success min 0.62 / median 0.94 / max 1.00; 18/20 searches found any dev improvement; 4/20 match or beat the Ripple patch on held-out success
- Structure-aware random search: random scale per output row (10 evals); mean of 20 independent searches: held-out success min 0.67 / median 1.00 / max 1.00; 19/20 searches found any dev improvement; 14/20 match or beat the Ripple patch on held-out success

Ripple patch improves held-out performance over the unpatched controller: **True**
