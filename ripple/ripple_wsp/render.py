import json
from pathlib import Path

BAR = "=" * 100


def show(run: Path, no_reasoning: bool = False, summary: bool = False) -> None:
    print(f"# run {run.name}\n")
    done = 0
    transcript = run / "transcript.jsonl"
    for line in transcript.read_text().splitlines() if transcript.exists() else []:
        t = json.loads(line)
        done = t["turn"]
        llm, acc = t["llm"], t.get("split_access") or {}
        split = f"{t['split_score']['accuracy']:.4f}" if t["model_status"] == "accepted" else t["model_status"]
        head = (
            f"TURN {t['turn']}  model.py {split}  split reads load={acc.get('load_split')} eval={acc.get('evaluate')}  "
            f"{llm.get('provider')} {llm.get('completion_tokens')} tok ({llm.get('reasoning_tokens')} reasoning) ${llm.get('cost') or 0:.4f}"
        )
        if summary:
            print(head)
            continue
        print(f"{BAR}\n{head}\n{BAR}")
        if t.get("reasoning") and not no_reasoning:
            print(f"--- reasoning ---\n{t['reasoning'].strip()}\n")
        print(f"--- reply ---\n{t['content'].strip()}\n")
        print(f"--- lab feedback ---\n{t['feedback'].strip()}\n")
    results = run / "results.json"
    if results.exists():
        r = json.loads(results.read_text())
        print(f"{BAR}\nRESULTS (val and test were never shown to the agent)\n{BAR}")
        for t in r["turns"]:
            s = f"{t['split']:.4f}" if t.get("split") is not None else t["model_status"]
            v = f"{t['val']:.4f}" if isinstance(t.get("val"), float) else "-"
            h = f"{t['test']:.4f}" if isinstance(t.get("test"), float) else "-"
            print(f"turn {t['turn']:>2}  split {s:>9}  val {v}  test {h}")
        print(f"final submission: {r['submission']}\nbest by split:    {r['best_by_split']}\nbest by val:      {r['best_by_val']}")
        print(f"cost ${r['cost']:.4f}, providers {r['providers']}")
        return
    live = run / "live.txt"
    if live.exists() and not summary:
        print(f"{BAR}\nTURN {done + 1} (in progress; reasoning and reply as they stream)\n{BAR}\n{live.read_text()}")
