# Outputs: the run directory

`RUNS/<run_id>/`:

| Path | Contents |
|---|---|
| `config.json` | regime, data paths, LLM settings, limits, allowed layers, the exact first message (`task`) and system message |
| `transcript.jsonl` | one record per turn: reply, reasoning, token counts and cost, model status (`accepted`, `rejected`, `failed`, `missing`) and why, split score, split reads, copied numbers (`reasoned`), feedback |
| `turns/turn_XX/` | `reply.md`, `reasoning.md`, `analysis.py`, `analysis.log`, `model.py`, `lab_access.jsonl`, `feedback.md`, `net.pt` and `weights/` when accepted, `score_split.json`, and after the run `score_val.json` and `score_test.json`. `turn_00/` is the seed in seeded regimes |
| `workspace/` | the agent's working directory (`model.py`, `history/`, its own files) |
| `agent/` | file backend only: `in_NNN.md` and `out_NNN.md` |
| `live.txt` | the reply streaming in right now |
| `results.json` | written after the last turn (see below) |

`results.json`:

```json
{
  "turns": [{"turn": 1, "model_status": "accepted", "split": 0.31, "val": 0.30, "test": 0.30, "test_per_class": [], "numbers": 8026, "split_access": {}}],
  "submission":    {"turn": 15, "split": 0.39, "val": 0.37, "test": 0.38, "numbers": 8026},
  "best_by_split": {"turn": 11, "...": "..."},
  "best_by_val":   {"turn": 11, "...": "..."},
  "cost": 0.05, "reminders": 0, "providers": ["Prime Intellect"]
}
```

- `submission`: the last accepted network, which is what the agent submitted.
- `best_by_split`: what the agent itself would call its best.
- `best_by_val`: for choosing across turns or runs.

Report `submission`, or a network chosen by val. Report test once.

Re-score any saved network:

```bash
python -m ripple_wsp eval --net RUN/turns/turn_11/net.pt --data /data/cifar10 --sets val,test --out scores.json
```
