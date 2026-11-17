Held-out evaluation, setting D, 100 episodes, deadline 7.0 s, run runs/demo_live_20261113_200222

| Condition | Success | Mean capped time (s) | Mean final dist (m) | Evals | Changed params | Patch L2 | Patch bytes | API $ | Repair s |
|---|---|---|---|---|---|---|---|---|---|
| Original controller, intact environment | 100% | 4.06 | 0.138 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.0 |
| Original controller, disturbed environment | 66% | 5.85 | 0.267 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.0 |
| Ripple-patched controller, disturbed environment | 100% | 4.63 | 0.142 | 9 | 32 | 2.293 | 4309 | 0.0392 | 483.8 |
| Ripple-patched controller, intact environment | 100% | 2.53 | 0.129 | 0 | 32 | 2.293 | 0 | 0.0000 | 0.0 |
| Patch removed (undo), disturbed environment | 66% | 5.85 | 0.267 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.0 |
| Estimated motor-gain compensation (external action correction, weights unchanged) | 100% | 4.19 | 0.138 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.26 |
| Random weight search, all 386 params (hill-climb, 10 evals); mean of 20 independent searches | 87% | 5.06 | 0.239 | 10 | 366.7 | 2.121 | 0 | 0.0000 | 0.65 |
| Structure-aware random search: random scale per output row (10 evals); mean of 20 independent searches | 96% | 4.61 | 0.149 | 10 | 32.3 | 3.518 | 0 | 0.0000 | 0.64 |

- Patch removed (undo), disturbed environment: original checkpoint restored, sha256 match=True
- Estimated motor-gain compensation (external action correction, weights unchanged): gains estimated from the 5 observed trajectories: left=1.000, right=0.500; assumes the fault is a wheel gain
- Random weight search, all 386 params (hill-climb, 10 evals); mean of 20 independent searches: held-out success min 0.64 / median 0.94 / max 1.00; 19/20 searches found any dev improvement; 3/20 match or beat the Ripple patch on held-out success
- Structure-aware random search: random scale per output row (10 evals); mean of 20 independent searches: held-out success min 0.66 / median 1.00 / max 1.00; 19/20 searches found any dev improvement; 12/20 match or beat the Ripple patch on held-out success

Ripple patch improves held-out performance over the unpatched controller: **True**
