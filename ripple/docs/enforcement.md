# Enforcement: what is checked mechanically, and what is not

| Guard | Where | Catches | Does not catch |
|---|---|---|---|
| Landlock file and network rules | `runner.py`, `sandbox.py` | reading anything but the Python install, `ripple/` and the split; writing outside the workspace; any TCP | data placed inside `ripple/` (the harness refuses such paths) |
| torch locks | `guard.py` | gradients, autograd, `torch.func` transforms, optimizers, linear solvers (numpy and torch) | hand-written iterative fitting (e.g. a perceptron update loop in numpy) |
| static scans | `common.py`, `tasks.py` | the same, plus absolute paths, `..`, environment access, subprocesses, network packages, `lab` in `model.py`, file reads in `reasoned` `model.py` | obfuscated code; comments are ignored on purpose |
| network check | `net.py`, `export.py`, `score.py` | non-allowed layer types, overridden `forward`, hooks, wrong output shape, more than 25M numbers. Scoring loads `net.pt` with `weights_only=True`, so no agent code runs | nothing about how the weights were chosen |
| `lab.evaluate` limit | `lab/data/lab.py` | more than 5 evaluations per turn | the agent computing accuracy itself to compare more variants (rule 4 is instruction beyond `evaluate`) |
| split-read log | `lab/data/lab.py`, harness fd | every `load_split` and `evaluate`, reported to the agent and stored in `turns/turn_XX/lab_access.jsonl` | |
| held-out separation | harness | val and test are not readable by agent code, and are scored only after the last turn | selection by test if *you* pick networks by test afterwards |
| copy detector (`reasoned`) | `harness.copied_numbers` | `model.py` with 6 or more numbers (3+ significant digits) matching printed analysis output | rounded, quantised or sign-only copies; see [regimes/reasoned.md](regimes/reasoned.md) |
| workspace cap | harness | workspaces over 2 GB (largest files deleted, agent told) | |
| memory and time caps | `common.limit_memory`, timeouts | analysis over 64 GiB address space or 10 minutes, scoring over 5 minutes | |

Everything marked "instruction" needs a transcript audit before a number is reported. `python -m ripple_wsp log RUN_DIR`
prints every turn's reasoning, reply and feedback. `turns/turn_XX/` holds the exact `analysis.py`, `analysis.log` and
`model.py`.
