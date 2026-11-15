# Weight-space programming with Ripple-65B

Hackathon project, Ricursive Intelligence hackathon, 13 November 2026 (US Pacific time).

**Question:** can [Ripple-65B Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) do *weight-space programming*
(write or edit neural-network weights directly, by reasoning, with no gradient training), and in the end edit **its own** weights?

**Short answer, from what we measured:**

- Ripple can write and repair tiny networks by hand (16/21 and 17/18), and repaired a robot controller's weights after a
  hidden motor fault, matching a conventional baseline on held-out episodes.
- Ripple edited **its own 65B weights** in two runs and the change was independently verified: it rewrote what it says the
  capital of France is (Paris → Berlin) and it tuned a pirate persona into itself.
- Neither self-edit is a self-*improvement*. The fact edit is a deliberate falsehood and one-way; the pirate edit costs
  answer correctness (9/32 answers still correct). Ripple's uppercase self-edit never produced an edit.
- Ripple's written reports about its own work overstate it. Every number in this repo was re-measured by us unless marked.

Who did what: "Fable" below is the agent that ran this project (Claude Fable 5.2, in Lor). Fable built the toolkit, ran
the GPU, wrote its own reference edits, and evaluated everything. "Ripple" is `ricursiveai/ripple-65b-preview`, called
through OpenRouter, acting as the editor in the experiments labelled as Ripple's.

> [!IMPORTANT]
> Early system-prompt scores in this project were wrong (see [docs/06-evaluation-and-errata.md](docs/06-evaluation-and-errata.md)).
> All numbers in this README use the corrected evaluation: replies run to completion (2,500-token cap), only the text
> after the reasoning block `</think>` is scored, and every condition is scored on the same 32 held-out test questions.

## Results at a glance

### 1. Ripple as a weight programmer (Ripple writes weights for other networks)

| Experiment | Result | Details |
|---|---|---|
| P1: write a tiny ReLU MLP for a function (abs, clamp, max, XOR, sort, band, median-of-3), one reply, no tools | 16 / 21 passed | 2 failures were malformed shapes; median-of-3 failed 3/3 (ran out of a 32k-token budget) |
| P2: find and fix one corrupted weight in an obfuscated tiny net | 17 / 18 passed | changed exactly 1 weight every time |
| P3: write MinAtar game policies from scratch (weight-space-bench) | near zero: best of 3 attempts Breakout 1.1, Seaquest 0.5, Asterix 0.7 (PPO-500M: 25.3, 88.9, 55.2) | per-attempt scores in [`minatar/README.md`](minatar/README.md); log `results/programmer/results_p3.log` |
| Ripple Repair: fix a rover controller after a hidden motor fault (5 settings) | 100% on 4 of 5, 88% on 1, on 100 held-out episodes | diagnosis exact in 3/5; random search with the same budget is nearly as good |

Docs: [docs/02-ripple-as-programmer.md](docs/02-ripple-as-programmer.md), [docs/05-ripple-repair.md](docs/05-ripple-repair.md).

### 2. A system prompt compiled into Ripple's weights (edits made by Fable)

No system prompt is given to the edited model. 32 held-out questions. "Correct" = judged correct by unedited Ripple with
style ignored.

| Behaviour | Condition | Behaviour shown | Content correct | Both |
|---|---|---|---|---|
| Uppercase | unedited | 0 | 32 | 0 |
| Uppercase | with the system prompt | 32 | 30 | 30 |
| Uppercase | edit v1 (built from cut-off reasoning text) | 12 | 26 | 10 |
| Uppercase | **edit v2 (built from answer tokens only)** | **27** | 28 | **27** |
| Pirate | unedited | 0 | 32 | 0 |
| Pirate | with the system prompt | 30 | 31 | 29 |
| Pirate | edit by Fable (strength 2.0) | 31 | 9 | 8 |
| Pirate | edit by Ripple itself (strength 2.63) | 31 | 9 | 8 |

Uppercase v2: 941,420 parameters, 8 always-on MLP neurons in each of 23 layers (34–78). Pirate: 655,114 / 655,203 parameters,
8 neurons in each of 16 layers (34–64). Terse, haiku and magic-conch prompts were only measured with 400-token replies
(terse partly works, haiku and conch did not). Docs: [docs/03-system-prompts-in-weights.md](docs/03-system-prompts-in-weights.md).

### 3. Ripple edits Ripple

| Run | Task | What Ripple was given | Outcome |
|---|---|---|---|
| c1 Spanish r1 | reply in Spanish | toolkit + Fable's findings | no edit; wrote a final report with invented numbers at step 7 |
| c1 Spanish r2 | same, plus a guard rejecting reports without saved files | | no edit; stopped emitting tool calls (provider PrimeIntellect) |
| c1 Spanish r3 | same, provider pinned to GMICloud | | killed at step 10 to switch task; no edit yet |
| **c1 pirate** | talk like a pirate | Fable's method (`persona_vectors`, `bake`) + which layers worked | **edit made**: Ripple swept 32 settings and chose strength 2.63 |
| **c2 fact** | capital of France → Berlin | write primitive + measurements only, no findings | **edit made**: Ripple searched ~55 settings, chose layers 48/52/56 |
| c3 uppercase r1, r1b | uppercase only | primitives, answer-only helper, told what failed before | no edit scored; r1 broken by the host Mac sleeping |

Fact edit, verified by Fable: P(next word = Berlin) 0% → 99% on the 4 prompts it built from, 68% on 5 rewordings it could
test on, **37% on 7 rewordings hidden from it**. The edit is one-way: "Paris is the capital of…" still completes to France
(99% before and after), and in chat only 1 of 4 France-capital questions answered Berlin (2 looped in reasoning).
Exact change: rank-one update of `mlp.down_proj.weight` in layers 48, 52, 56, 128 columns each, 1,935,075 changed values.

Docs: [docs/04-ripple-edits-ripple.md](docs/04-ripple-edits-ripple.md).

## Repository map

| Path | What it is |
|---|---|
| `docs/` | one write-up per experiment, plus errata and the harness |
| `code/toolkit/` | `lab.py` (load, generate, capture activations, edit/rollback/export), `wsp1.py` (persona directions, `bake`), `fact.py` (rank-one fact writes), `kernel.py` (persistent GPU kernel) |
| `code/infra/` | `vm.py` / `xfer.py`: drive the GPU virtual machine and move files through it |
| `code/programmer/prog.py` | P1/P2 tiny-network tasks and scorer |
| `code/evalkit/` | queries (48 train / 16 val / 32 test), the 6 system prompts and their checks, the API judge |
| `code/experiments/` | every script that was run in the GPU kernel, in the order listed in `docs/07-harness.md` |
| `code/analysis/summarize.py` | recomputes every table in this README from `results/` |
| `results/` | raw replies and scores; `superseded_64tok/` holds the early, wrong-method runs |
| `patches/` | saved weight edits (see `patches/README.md`) |
| `self_edit_runs/` | briefings, task prompts and full transcripts of Ripple's self-edit runs |
| `deck/` | the presentation (`deck.html`) and the script that builds it from measured numbers |
| `techniques/catalogue.md` | survey of published weight-space-programming techniques, with links |
| `agent/`, `ripple-repair/`, `minatar/` | the Ripple agent loop, the rover repair project, the MinAtar benchmark runs (pushed from the second machine) |
| `self_edit_runs/transcripts/` | every agent run directory and log as written on the Mac (`agent_runs/`), including Spanish, toy and misfire runs |
| `results/other/` | early logs: P2 first pass with HTTP 402 errors (`results_p2b.log`), judge sanity check, pirate validation replies and scores |

## Cost

- GPU: one-B300 virtual machines on Lor, about $80 in total by our arithmetic from the listed rate (~$9/hour); not checked
  against the bill. One machine sat idle for about an hour before it expired.
- Ripple API (OpenRouter): cents per run; every transcript records `usage.cost` per call. Ripple Repair logged $0.47.

## Known gaps

- The pirate edit rebuilt with the uppercase-v2 recipe was started but lost when its machine expired; not measured.
- Ripple's uppercase self-edit never produced an edit; a fair rerun with answer-only directions is still open.
- One run per setting throughout; no variance estimates.
- The correctness judge is unedited Ripple itself (YES/NO probability); not calibrated against humans.
