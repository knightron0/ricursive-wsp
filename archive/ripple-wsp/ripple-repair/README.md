# Ripple Repair

An LLM (Ripple, `ricursiveai/ripple-65b-preview`) repairs a simulated rover's small neural
controller by **directly editing its weights**, with no gradient training in the repair phase.

Ripple is the weight-space programmer; the rover's 4-16-16-2 tanh network (386 parameters) is the
editable target. Ripple does not modify itself, and this toy does not show general robotic adaptation.

## Result in one table (setting A: left motor at half strength, hidden from Ripple)

100 held-out episodes, evaluated once, after the patch was selected on 20 development episodes.
Task: reach within 0.15 m of the goal inside 7 s.

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

- **Ripple's diagnosis matched the hidden disturbance.** Hidden: `left_gain = 0.5`. Ripple, from five
  trajectories: "left motor's gain changed from 1.0 to 0.5", checked step by step against the
  simulator equations (`runs/A_r1/final.md`).
- **What the patch changed:** 17 of 386 parameters, all in the output layer: row 0 of `W3` (the 16
  weights into the left-wheel output) multiplied by 6, and `b3[0]` written back as its rounded
  original value (-0.0674, a 5e-5 change). The left wheel's command is pushed toward the +/-1 limit
  so the half-strength motor delivers about the wheel speed the controller intended; the right
  wheel is untouched. Evidence for the mechanism: the estimated-gain baseline (divide the left
  command by the fitted gain 0.500) gives the same recovery, and Ripple's own sweep over scale
  factors 2, 2.5, 3, 4, 6, 10 improved up to about 6 and was flat beyond (`runs/A_r1/candidates.jsonl`).
- **Undo:** removing the patch gives back the original checkpoint byte for byte (same sha256) and the
  degraded 67%.
- **Caveat, generic search also works here.** An equal-budget random search over all 386 weights
  (10 evaluations, same 20 dev episodes, same ranking rule) recovers part of the loss on average,
  and a random search that only scales the two output rows usually finds a full repair. The
  controller cruises at half throttle, so many perturbations that raise the weak wheel's output
  help. What Ripple adds over random search in this run is a correct diagnosis, a 17-parameter
  explainable edit, and 100% on the first candidate; it is not the only way to find a working
  patch, and in two of the five runs below its diagnosis was not specific.
- **Caveat, the failure exists only under a deadline.** See "Viability" below.

## All five repair runs (held-out, 100 episodes each, success / mean capped time)

| Setting | Hidden gain | Original, damaged | Ripple patch, damaged | Ripple patch, intact env | Estimated-gain compensation (external) | Random search, all weights (mean of 20) | Random search, output-row scales (mean of 20) | Ripple's diagnosis | Matches hidden? | Best patch | Evals | API cost | Time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | left 0.5 | 67% / 5.81 s | **100% / 4.17 s** | 100% / 3.58 s | 100% / 4.21 s | 88% | 97% | "left motor's gain changed from 1.0 to 0.5" | yes, exact | W3 row 0 x6 (b3[0] kept) | 9 | $0.117 | 18 min |
| B | right 0.6 | 80% / 5.37 s | **100% / 3.52 s** | 100% / 3.31 s | 100% / 4.11 s | 93% | 99% | "right wheel's effective gain is ~0.60" | yes, exact | W3 row 1 and b3[1] x6 | 10 | $0.064 | 8 min |
| C | left 0.4 | 40% / 6.28 s | **88% / 5.19 s** | 100% / 3.06 s | 89% / 5.14 s | 68% | 80% | "left-wheel gain dropped from 1.0 to ~0.4" (first guess: both wheels slower) | yes, after 8 candidates | W3 row 0 and b3[0] x3.5 | 10 | $0.137 | 20 min |
| D | right 0.5 | 66% / 5.85 s | **100% / 4.63 s** | 100% / 2.53 s | 100% / 4.19 s | 87% | 96% | "reduced turn authority", no motor named | no (vague) | both W3 rows x1.8 | 9 | $0.039 | 8 min |
| E | left 0.7 | 93% / 4.90 s | **100% / 3.15 s** | 100% / 2.27 s | 100% / 4.08 s | 99% | 100% | "little turn authority", no motor named | no (vague) | all of W3 x2.5 | 5 | $0.031 | 5 min |

Original controller in the intact environment: 100% / 4.06 s on the same seeds.

- A, B, C were run from the command line; D and E are the two live runs started from the demo
  page (slider: right motor 50%, left motor 30%). All five are complete runs and all five are
  reported; none was dropped. Failed starts with zero evaluated candidates (a harness problem,
  see below) are kept in `runs/aborted/`, `runs/B_r1`, `runs/B_r2`. Entries D and E in
  `hidden/disturbances.json` were added after those runs, so that `evaluate.py` can score them.
- **C is not a full repair:** 88%, level with the estimated-gain baseline (89%). The weak wheel
  saturates at gain 0.4, so the 7 s deadline cannot be met for the farthest goals this way; the
  best single random search reached 93%.
- **D and E show the limit of the diagnosis claim:** Ripple did not identify the motor in either.
  It scaled the whole output layer (x1.8, x2.5), which still gave 100% on held-out. The patches
  worked; the explanations were not specific. Diagnosis matched the hidden fault in 3 of 5 runs.
  E is also a mild fault (93% before repair) and Ripple stopped after 5 evaluations.
- The patched controllers are *faster* than the original in the intact environment (they
  over-drive a wheel relative to the teacher's half-throttle cruise) and still reach 100%. Intact
  performance was not hurt on these metrics; path shape and smoothness were not measured.
- In B the patched damaged rover (3.52 s) is faster than the original intact rover (4.06 s) for
  the same reason: the patch saturates the weak wheel, which then delivers 0.6 m/s, above the
  0.5 m/s cruise. "Restored" here means the success and time metrics, not identical behaviour.

All repair runs above are Ripple's autonomous result. There is no developer-assisted repair in this
repository. (One developer feasibility check exists, `scale W3 row x3`, run on dev seeds during the
viability stage to confirm that some weight-space repair exists; it was never shown to Ripple and is
not a reported result.)

## What is in the folder

| File | Role |
|---|---|
| `sim.py`, `config.json` | Deterministic simulator, the controller forward pass, the teacher; frozen settings |
| `train_controller.py` | Setup training of the original controller (the only gradient use) |
| `checkpoints/original.json` (+`.sha256`) | Original controller, sha256 `bc0d785c...b833cc` |
| `viability.py`, `logs/viability.{txt,json}` | Section-4 sweep |
| `repair.py`, `ripple_api.py` | Ripple repair harness: five JSON tools, patch validation, cost ledger and cap |
| `evaluate.py` | Held-out evaluation and baselines |
| `splits.json`, `hidden/` | Seed splits; `hidden/` holds the disturbance settings and held-out seeds |
| `runs/<setting>_r1/` | Per run: `transcript.jsonl`, `candidates.jsonl`, `best_patch.json`, `patched.json`, `result.json`, `final.md` |
| `results/heldout_*.{json,md}` | Held-out tables |
| `demo/` | Browser demo (`server.py`, `index.html`) and the recorded backup (`recorded/`) |
| `logs/api_cost.jsonl` | `usage.cost` of every API response |

## Setup

- **Simulator** (`sim.py`): differential drive exactly as specified
  (`left_velocity = max_speed * left_gain * left_command`, ...), dt 0.05 s, wheel separation 0.4 m,
  max wheel speed 1 m/s, success radius 0.15 m, goal 1-3 m away in a random direction, random
  heading, no obstacles.
- **Teacher**: bearing `a = atan2(dy, dx)`; forward command `0.5 * max(0, cos a)` (turns in place
  when facing away); steering `clip(1.0 * a, +/-0.5)`; `left = v - s`, `right = v + s`; then
  low-pass with the previous command (`0.7 * target + 0.3 * previous`).
- **Controller**: 4 inputs (goal dx/3, goal dy/3 in rover coordinates, previous left and right
  command) -> 16 tanh -> 16 tanh -> 2 tanh. **Setup training** (documented, gradients allowed
  here only): supervised regression to the teacher on 60k uniformly sampled inputs plus states
  visited by the network itself in intact episodes (4 rounds), hand-written backprop + Adam in
  numpy, about 80 s of CPU. Final imitation MSE 3.3e-4. Closed-loop validation: 100% on 100
  setup-validation episodes and on the 100 development episodes (mean 4.06 s, same as the teacher).

## Viability (section 4): motor weakening alone does not break the controller

Measured on 100 development episodes (`logs/viability.txt`):

| Left motor gain | Success at 20 s | Success at 7 s | Mean time (s) | With conventional compensation, success at 7 s |
|---|---|---|---|---|
| 1.0 (intact) | 100% | 100% | 4.06 | – |
| 0.7 | 100% | 90% | 4.92 | 100% |
| 0.6 | 100% | 78% | 5.43 | 100% |
| 0.5 | 100% | 63% | 6.16 | 100% |
| 0.4 | 100% | 50% | 7.30 | 85% |
| 0.3 | 100% | 25% | 9.26 | 55% |

(The right motor behaves the same within a few points.)

**Finding:** with the suggested 20 s episode, weakening either motor to 30% never reduces success
(97% at 20%). The controller is closed-loop on the goal bearing, so it steers against the
imbalance and still arrives, only slower (+52% time at gain 0.5). Motor weakening is a *speed*
fault for this controller, not a *reachability* fault.

**Decision, taken before any Ripple call and before touching held-out seeds:** keep motor weakening
as the disturbance and make the deadline binding: episode duration 20 s -> **7 s** (1.12x the
slowest intact development episode, 6.25 s). Under that deadline gain 0.5 fails 37% of
development episodes, and dividing the weak motor's command by its gain (conventional
compensation) restores 100%, so the failure is repeatable and physically recoverable. At gain 0.4
full recovery is impossible (the weak wheel saturates): the ceiling with oracle compensation is 85%.
The config was then frozen (`config.json`, `"frozen": true`). No alternative disturbance type was
needed. The 20 s numbers for the final patch are in `results/heldout_A_supp_T20.md`: there the
patch changes time (6.18 s -> 4.17 s), not success (100% -> 100%).

## Repair interface (sections 5-6)

Ripple receives: architecture and input/output meanings, the nominal simulator equations, the
original weights (`inspect_weights`), five trajectories of the original controller in the changed
environment (`get_trajectory`: time, x, y, heading, goal, network inputs, commands), and the
statement that "an actuator or sensor property has changed". It does not receive the disturbance
type or value, simulator internals, held-out seeds, or any repaired checkpoint. The hidden setting
lives in `hidden/disturbances.json`; `repair.py` passes it only into the simulator call and no tool
output contains it. Ripple has no shell and no file access: only
`inspect_weights, get_trajectory, submit_patch, evaluate_candidate, restore_original`.

A patch is a list of numeric edits to existing tensors, each `set`, `add` or `scale` on one
element, row, column or whole tensor, always applied to the original checkpoint. Shapes are
fixed; non-finite or out-of-range values are rejected. Each `evaluate_candidate` runs the fixed 20
development episodes, returns aggregate metrics plus the worst, median and best trajectory, and
uses one of 10 evaluations. Ranking: success rate first, then lower mean capped completion time
(failures count as 7 s). The best candidate is saved as `best_patch.json` (sparse old/new values)
separately from the original.

Harness rules that are not in the task text, all generic (no hint about the fault or the fix):

- A final answer with zero evaluated candidates is rejected and the model is told to continue.
- After each batch of tool results the harness adds a neutral user turn ("Continue. Evaluations
  left: N ... write in plain text what you conclude so far ..."). Without it this model/provider
  often returned an empty message at the token limit; after an empty reply the harness re-words
  the note and alternates a larger token budget. Details and the failed starts this caused are
  under "What did not work".
- Tool calls written as text or a bare JSON patch would be parsed and logged as such; in the runs
  here every tool call was structured (`result.json: tool_call_forms`).
- Provider pinned to GMICloud; 5xx responses are retried.

## Honest-evaluation notes (section 7)

- Held-out seeds 9000-9099 are read only by `evaluate.py`; every read is appended to
  `logs/heldout_access.jsonl`. Each setting was evaluated on held-out once, after its repair run
  ended (plus one supplementary pass for A at 20 s). `evaluate.py` was debugged on development
  seeds through `--seeds-file`.
- Same seeds for every condition.
- The gain-compensation baseline fits left/right gains by least squares to the same five observed
  trajectories (it assumes the fault is a wheel gain, which Ripple was not told) and corrects the
  action **outside** the network; the weights are unchanged. It is a different kind of
  implementation from a weight patch.
- Random search, two variants, 20 independent searches each: (a) hill-climb with Gaussian noise on
  all 386 parameters, sigma log-uniform in [0.01, 0.32]; (b) structure-aware: a random scale in
  [1/4, 4] on each output row. Both get the original's dev score for free, as Ripple does.
- The patched controller is also run in the intact environment (row 4 of the table).

## Reproduce

```bash
cd ~/Developer/ripple-repair            # Python 3.11 + numpy, nothing else
python3 train_controller.py             # optional: rebuilds checkpoints/original.json (same sha256)
python3 viability.py --durations 20 7   # section-4 sweep -> logs/viability.json

export OPENROUTER_API_KEY=...           # Ripple via OpenRouter
python3 repair.py --setting A --out runs/A_r2      # a new Ripple repair (not deterministic), ~$0.04-0.14, 8-20 min
python3 evaluate.py --setting A --run runs/A_r1    # held-out table for the saved run -> results/heldout_A.{json,md}
python3 evaluate.py --setting B --run runs/B_r3
python3 evaluate.py --setting C --run runs/C_r1
python3 evaluate.py --setting D --run runs/demo_live_20261113_200222
python3 evaluate.py --setting E --run runs/demo_live_20261113_203133

python3 demo/server.py                  # then open http://127.0.0.1:8765
python3 demo/make_recorded.py --run runs/A_r1 --setting A   # refresh the recorded backup
```

`evaluate.py` and `viability.py` are deterministic and make no API calls. `repair.py` refuses to
call the API once `logs/api_cost.jsonl` reaches `RIPPLE_SPEND_CAP_USD` (default 15).

## Demo

`python3 demo/server.py`, then http://127.0.0.1:8765.

- Rover and goal, with intact / damaged / repaired playback from the same start state.
- Motor-damage slider (and left/right choice). The value goes to the simulator, not to Ripple.
- "Diagnose and repair" starts a live Ripple run for the current slider setting; the log shows its
  hypotheses, each patch and each score as they arrive; the best-so-far patch becomes playable as
  soon as it beats the original. A full run takes 8-20 minutes.
- Changing the slider drops the patch: a new setting needs a new repair. Nothing cached is shown as live.
- "Undo patch" restores the original checkpoint. Scoreboard: 50 demo episodes (seeds 4000-4049, not held-out).
- All wheel commands come from the small network in `sim.py`; Ripple is called only during repair.
- **Recorded backup:** "Load recorded run" replays run `runs/A_r1` (log and patch) with a RECORDED
  label. Opening `demo/index.html` directly as a file (no server) shows the same recorded run,
  also labelled RECORDED, with eight precomputed start states.

## Cost

| Item | Cost |
|---|---|
| Run A (`runs/A_r1`) | $0.117 |
| Run B (`runs/B_r3`) | $0.064 |
| Run C (`runs/C_r1`) | $0.137 |
| Run D, live from the demo (`runs/demo_live_20261113_200222`) | $0.039 |
| Run E, live from the demo (`runs/demo_live_20261113_203133`) | $0.031 |
| Failed starts with 0 evaluations (A attempt 0/1, `B_r1`, `B_r2`) | $0.078 |
| Logged probes | $0.005 |
| **Ledger total (`logs/api_cost.jsonl`)** | **$0.47** |
| Unlogged debugging probes of the empty-reply problem (about 70 calls sent outside `ripple_api.py`) | about $0.1, estimated, not measured |

Ripple price on OpenRouter at the time: $0.10 / M input tokens, $0.30 / M output tokens. A full
repair run costs $0.03-0.14 and takes 5-20 minutes. No GPU was used; all simulation and
setup training ran on a laptop CPU. Cap for this build: $15 (enforced in `ripple_api.py`).

## What did not work / limits

- **Motor weakening does not break this controller at 20 s** (see Viability). The failure is a
  deadline failure created by freezing the episode length at 7 s. That choice was made on
  development data before any repair, but it is a design choice and the result depends on it.
- **Random search is a strong baseline here.** Averaged over 20 searches it is below Ripple in
  every setting, but individual searches often match it (table above, and the per-search numbers
  in `results/heldout_*.json`). The repair is easy: a one-parameter family (scale the weak wheel's
  output row) contains a full fix, and even scaling everything up helps.
- **Hidden reasoning past the token limit.** After tool results Ripple often spends its whole
  output budget on reasoning that the provider strips, returning an empty message. This killed one
  attempt at A (16k tokens, empty) and two at B (`runs/B_r1`, `runs/B_r2`), each before any
  candidate was evaluated. The harness now adds a neutral "Continue ... write in plain text what
  you conclude" turn after tool results and, after an empty reply, alternates a 16-24k token budget
  with a re-worded note. Runs A_r1 and C_r1 used the first version of that note, D the second,
  B_r3 and E the final one; `transcript.jsonl` in each run records the exact messages.
- **A report with no evaluation was attempted once** (run D, 196 s in) and rejected by the harness;
  Ripple then submitted and evaluated real patches.
- **Provider instability:** GMICloud returned bursts of immediate 502/503 errors; the client
  retries up to 40 times.
- **Patches use `scale` edits.** Ripple mostly wrote multiplicative edits ("W3 row 0 x6") that the
  harness expands into numbers; `best_patch.json` lists every changed weight with old and new
  value. It never rewrote hidden-layer weights.
- **One disturbance family only** (a single wheel gain, five settings, one repair run each). No
  sensor faults, no dead zones, no obstacles, no noise, one controller. Run-to-run variance of
  Ripple was not measured.
- Live repair in the demo takes 8-20 minutes; the recorded run is the practical choice on stage.
