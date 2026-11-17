Held-out evaluation, setting C, 100 episodes, deadline 7.0 s, run runs/C_r1

| Condition | Success | Mean capped time (s) | Mean final dist (m) | Evals | Changed params | Patch L2 | Patch bytes | API $ | Repair s |
|---|---|---|---|---|---|---|---|---|---|
| Original controller, intact environment | 100% | 4.06 | 0.138 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.0 |
| Original controller, disturbed environment | 40% | 6.28 | 0.409 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.0 |
| Ripple-patched controller, disturbed environment | 88% | 5.19 | 0.171 | 10 | 17 | 5.006 | 2695 | 0.1367 | 1205.4 |
| Ripple-patched controller, intact environment | 100% | 3.06 | 0.141 | 0 | 17 | 5.006 | 0 | 0.0000 | 0.0 |
| Patch removed (undo), disturbed environment | 40% | 6.28 | 0.409 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.0 |
| Estimated motor-gain compensation (external action correction, weights unchanged) | 89% | 5.14 | 0.167 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.21 |
| Random weight search, all 386 params (hill-climb, 10 evals); mean of 20 independent searches | 68% | 5.67 | 0.272 | 10 | 347.4 | 3.015 | 0 | 0.0000 | 0.67 |
| Structure-aware random search: random scale per output row (10 evals); mean of 20 independent searches | 80% | 5.40 | 0.196 | 10 | 32.3 | 3.642 | 0 | 0.0000 | 1.0 |

- Patch removed (undo), disturbed environment: original checkpoint restored, sha256 match=True
- Estimated motor-gain compensation (external action correction, weights unchanged): gains estimated from the 5 observed trajectories: left=0.400, right=1.000; assumes the fault is a wheel gain
- Random weight search, all 386 params (hill-climb, 10 evals); mean of 20 independent searches: held-out success min 0.33 / median 0.72 / max 0.93; 18/20 searches found any dev improvement; 4/20 match or beat the Ripple patch on held-out success
- Structure-aware random search: random scale per output row (10 evals); mean of 20 independent searches: held-out success min 0.40 / median 0.83 / max 0.88; 19/20 searches found any dev improvement; 1/20 match or beat the Ripple patch on held-out success

Ripple patch improves held-out performance over the unpatched controller: **True**
