Held-out evaluation, setting E, 100 episodes, deadline 7.0 s, run runs/demo_live_20261113_203133

| Condition | Success | Mean capped time (s) | Mean final dist (m) | Evals | Changed params | Patch L2 | Patch bytes | API $ | Repair s |
|---|---|---|---|---|---|---|---|---|---|
| Original controller, intact environment | 100% | 4.06 | 0.138 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.0 |
| Original controller, disturbed environment | 93% | 4.90 | 0.149 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.0 |
| Ripple-patched controller, disturbed environment | 100% | 3.15 | 0.138 | 5 | 32 | 4.299 | 4307 | 0.0313 | 280.3 |
| Ripple-patched controller, intact environment | 100% | 2.27 | 0.128 | 0 | 32 | 4.299 | 0 | 0.0000 | 0.0 |
| Patch removed (undo), disturbed environment | 93% | 4.90 | 0.149 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.0 |
| Estimated motor-gain compensation (external action correction, weights unchanged) | 100% | 4.08 | 0.138 | 0 | 0 | 0.000 | 0 | 0.0000 | 0.25 |
| Random weight search, all 386 params (hill-climb, 10 evals); mean of 20 independent searches | 99% | 3.93 | 0.142 | 10 | 366.7 | 2.397 | 0 | 0.0000 | 0.59 |
| Structure-aware random search: random scale per output row (10 evals); mean of 20 independent searches | 100% | 3.43 | 0.139 | 10 | 32.3 | 3.867 | 0 | 0.0000 | 1.09 |

- Patch removed (undo), disturbed environment: original checkpoint restored, sha256 match=True
- Estimated motor-gain compensation (external action correction, weights unchanged): gains estimated from the 5 observed trajectories: left=0.700, right=1.000; assumes the fault is a wheel gain
- Random weight search, all 386 params (hill-climb, 10 evals); mean of 20 independent searches: held-out success min 0.90 / median 1.00 / max 1.00; 19/20 searches found any dev improvement; 13/20 match or beat the Ripple patch on held-out success
- Structure-aware random search: random scale per output row (10 evals); mean of 20 independent searches: held-out success min 0.93 / median 1.00 / max 1.00; 19/20 searches found any dev improvement; 19/20 match or beat the Ripple patch on held-out success

Ripple patch improves held-out performance over the unpatched controller: **True**
