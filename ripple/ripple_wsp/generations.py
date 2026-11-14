import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ripple_wsp.cli import run_id


def candidates(run: Path) -> list[dict]:
    res = run / "results.json"
    if not res.exists():
        return []
    return [
        {"run": run.name, "turn": t["turn"], "split": t["split"], "val": t["val"], "test": t["test"]}
        for t in json.loads(res.read_text())["turns"]
        if t["turn"] > 0 and isinstance(t.get("val"), float)
    ]


def save(td: Path, dest: Path, note: str) -> None:
    dest.mkdir(parents=True)
    for name in ("model.py", "net.pt"):
        shutil.copy2(td / name, dest / name)
    if (td / "weights").is_dir():
        shutil.copytree(td / "weights", dest / "weights")
    (dest / "README.txt").write_text(note + "\n")


def seed_val(seed: Path, data: str) -> float:
    out = seed / "val.json"
    if not out.exists():
        subprocess.run([sys.executable, "-m", "ripple_wsp", "eval", "--net", str(seed / "net.pt"), "--data", data, "--sets", "val", "--out", str(out)], check=True)
    return json.loads(out.read_text())["val"]["accuracy"]


def loop(regime: str, seed: Path, saved: Path, data: str, runs: str, generations: int, replicates: int, extra: list[str]) -> None:
    runs_dir = Path(runs).expanduser()
    log = saved / "generations.jsonl"
    saved.mkdir(parents=True, exist_ok=True)
    best = seed_val(seed, data)
    for gen in range(1, generations + 1):
        ids = [run_id() for _ in range(replicates)]
        procs = [
            subprocess.Popen([sys.executable, "-m", "ripple_wsp", "run", regime, "--data", data, "--runs", runs, "--seed", str(seed), "--run-id", i, *extra],
                             stdout=(runs_dir / f"{i}.log").open("w"), stderr=subprocess.STDOUT)
            for i in ids
        ]
        print(f"gen {gen}: seed {seed} (val {best:.4f}); runs {ids}", flush=True)
        codes = [p.wait() for p in procs]
        cands = [c for i in ids for c in candidates(runs_dir / i)]
        top = max(cands, key=lambda c: c["val"], default=None)
        rec = {"gen": gen, "seed": str(seed), "seed_val": best, "runs": ids, "exit_codes": codes, "top": top, "n_candidates": len(cands)}
        if top and top["val"] > best:
            td = runs_dir / top["run"] / "turns" / f"turn_{top['turn']:02d}"
            dest = saved / f"gen{gen}_val{top['val'] * 100:.1f}_{time.strftime('%Y%m%d_%H%M%S')}"
            save(td, dest, f"gen {gen}: run {top['run']} turn {top['turn']}, split {top['split']:.4f}, val {top['val']:.4f}, test {top['test']:.4f}; seeded from {seed}")
            (dest / "val.json").write_text(json.dumps({"val": {"accuracy": top["val"]}}) + "\n")
            seed, best = dest, top["val"]
            rec["new_seed"] = str(dest)
        with log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"gen {gen} done: top {top}; seed now {seed} (val {best:.4f})", flush=True)
