import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import tokenize
from pathlib import Path

from ripple_wsp.common import BANNED, CLASSES, FIT_BANNED, compress_tracebacks, head_tail, limit_memory
from ripple_wsp.llm import REMOTE_MODEL, FileAgent, OpenRouter
from ripple_wsp.tasks import COPY_LIMIT, MODEL_BANNED, MODEL_FILE_BANNED, REMINDER, SYSTEM, TASKS

ROOT = Path(__file__).resolve().parents[1]
LAB = Path(__file__).with_name("lab")
BLOCK = re.compile(r"```(?:python|py)[ \t]*\n(.*?)(?:```|\Z)", re.S)
MODEL_TAG = re.compile(r"\A\s*#\s*model\.py\b")

def strip_comments(code: str) -> str:
    lines = code.splitlines(keepends=True)
    try:
        comments = [t for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type == tokenize.COMMENT]
    except (tokenize.TokenError, SyntaxError):
        return code
    for t in comments:
        (row, start), (_, end) = t.start, t.end
        line = lines[row - 1]
        lines[row - 1] = line[:start] + " " * (end - start) + line[end:]
    return "".join(lines)


def violations(code: str, rules: list[tuple[str, str]]) -> list[str]:
    code = strip_comments(code)
    return [why for pat, why in rules if re.search(pat, code)]


def parse(content: str) -> tuple[str | None, str | None, int]:
    blocks = BLOCK.findall(content)
    models = [b for b in blocks if MODEL_TAG.match(b)]
    analysis = [b for b in blocks if not MODEL_TAG.match(b)]
    code = "\n\n".join(analysis) if analysis else None
    return code, (models[-1] if models else None), len(blocks)


def plan_of(content: str) -> str:
    return BLOCK.sub("", content).strip()


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


WS_CAP = 2 * 2**30


def enforce_cap(ws: Path) -> list[str]:
    files = sorted((p for p in ws.rglob("*") if p.is_file() and p.name != "model.py" and "history" not in p.relative_to(ws).parts), key=lambda p: -p.stat().st_size)
    total, removed = dir_bytes(ws), []
    for p in files:
        if total <= WS_CAP:
            break
        size = p.stat().st_size
        p.unlink()
        total -= size
        removed.append(f"{p.relative_to(ws)} ({size / 2**30:.1f} GB)")
    return removed


NUM = re.compile(r"(?<![\w.])-?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?")


def numbers(text: str) -> list[tuple[float, int]]:
    out = []
    for s in NUM.findall(text):
        mant = s.lstrip("-").lower().split("e")[0]
        digits = mant.replace(".", "").lstrip("0")
        try:
            out.append((float(s), len(digits), len(mant.split(".")[1]) if "." in mant else 0))
        except ValueError:
            pass
    return out


def copied_numbers(model_code: str, printed: list[str], own: list[str]) -> list[str]:
    own_vals = {round(v, 6) for text in own for v, _, _ in numbers(strip_comments(text))}
    seen = sorted({v for text in printed for v, _, _ in numbers(text) if round(v, 6) not in own_vals and v != 0})
    import bisect

    hits = []
    for v, digits, decimals in numbers(strip_comments(model_code)):
        if digits < 3 or round(v, 6) in own_vals:
            continue
        tol = 0.5 * 10 ** (-decimals) + 1e-12
        i = bisect.bisect_left(seen, v - tol)
        if i < len(seen) and seen[i] <= v + tol:
            hits.append(f"{v:g}")
    return hits


def run(cmd: list[str], cwd: Path, env: dict, timeout: int, log: Path, pass_fds: tuple = ()) -> tuple[int | None, float]:
    t0 = time.time()
    with log.open("w") as f:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, preexec_fn=limit_memory, pass_fds=pass_fds)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            rc = None
    return rc, time.time() - t0


def failure(rc: int | None, timeout: int, log: Path) -> str:
    if rc is None:
        return f"timed out after {timeout}s"
    text = compress_tracebacks(log.read_text(errors="replace")).strip()
    return text[-1500:] if len(text) > 1500 else text


def export(model_code: str, weights: Path, td: Path, env: dict, timeout: int) -> dict:
    box = td / "box"
    shutil.rmtree(box, ignore_errors=True)
    box.mkdir(parents=True)
    (box / "model.py").write_text(model_code)
    if weights.is_dir():
        shutil.copytree(weights, box / "weights")
    rc, secs = run([sys.executable, "-m", "ripple_wsp.export"], box, env, timeout, td / "export.log")
    net = box / "net.pt"
    if rc != 0 or not net.exists():
        shutil.rmtree(box, ignore_errors=True)
        return {"error": f"building the network failed: {failure(rc, timeout, td / 'export.log')}"}
    info = json.loads((box / "export.json").read_text())
    shutil.move(net, td / "net.pt")
    if (box / "weights").is_dir():
        shutil.rmtree(td / "weights", ignore_errors=True)
        shutil.move(box / "weights", td / "weights")
    shutil.rmtree(box, ignore_errors=True)
    return info


def score(net: Path, data: Path, where: Path, env: dict, timeout: int) -> dict:
    box = where.with_suffix(".box")
    shutil.rmtree(box, ignore_errors=True)
    box.mkdir(parents=True)
    shutil.copy2(net, box / "net.pt")
    out, log = where.with_suffix(".json"), where.with_suffix(".log")
    try:
        rc, secs = run([sys.executable, "-m", "ripple_wsp.score", str(box / "net.pt"), str(data), str(out)], box, env, timeout, log)
    finally:
        shutil.rmtree(box, ignore_errors=True)
    if rc != 0 or not out.exists() or not out.stat().st_size:
        return {"error": f"running the network failed: {failure(rc, timeout, log)}"}
    return json.loads(out.read_text())


def score_text(r: dict) -> str:
    line = f"network: {r['numbers']:,} numbers; layers: {head_tail(r['layers'], 600)}"
    line += f"\nsplit accuracy {r['accuracy']:.4f} ({r['seconds']:.0f}s for {r['n']} images)"
    line += "\nper class: " + ", ".join(f"{c} {a:.2f}" for c, a in zip(CLASSES, r["per_class"]))
    short = [c[:5] for c in CLASSES]
    line += "\nconfusion matrix (rows = true class, columns = predicted, 500 images per row):\n"
    line += "       " + " ".join(f"{s:>5}" for s in short) + "\n"
    line += "\n".join(f"{short[i]:>6} " + " ".join(f"{v:5d}" for v in row) for i, row in enumerate(r["confusion"]))
    return line


def read_access(path: Path) -> dict:
    events = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    loads = sum(e["event"] == "load_split" for e in events)
    evals = [e["accuracy"] for e in events if e["event"] == "evaluate"]
    refused = sum(e["event"] == "evaluate_refused" for e in events)
    return {"load_split": loads, "evaluate": len(evals), "evaluate_accuracies": evals, "evaluate_refused": refused}


def access_text(a: dict, ran: bool) -> str:
    if not ran:
        return "Split reads this turn: none (no analysis code ran)."
    if not a["load_split"] and not a["evaluate"]:
        return "Split reads this turn: none. Your code did not read the split."
    text = f"Split reads this turn: lab.load_split x{a['load_split']}, lab.evaluate x{a['evaluate']}"
    if a["evaluate"]:
        text += " (" + ", ".join(f"{v:.4f}" for v in a["evaluate_accuracies"]) + ")"
    if a["evaluate_refused"]:
        text += f", {a['evaluate_refused']} evaluate call(s) refused over the limit"
    return text + "."


def build_messages(task: str, turns: list[dict], keep_full: int) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    for i, t in enumerate(turns):
        if i >= len(turns) - keep_full:
            msgs += [{"role": "assistant", "content": t["content"]}, {"role": "user", "content": t["feedback"]}]
        else:
            note = f"[turn {t['turn']}: code omitted from history; saved in the workspace as history/turn_{t['turn']:02d}_analysis.py and history/turn_{t['turn']:02d}_model.py]"
            msgs += [{"role": "assistant", "content": f"{plan_of(t['content'])[:1500]}\n\n{note}"}, {"role": "user", "content": t["feedback_short"]}]
    return msgs


def fit_context(task: str, turns: list[dict], budget_chars: int) -> tuple[list[dict], int]:
    keep = len(turns)
    msgs = build_messages(task, turns, keep)
    while keep > 1 and sum(len(m["content"]) for m in msgs) > budget_chars:
        keep -= 1
        msgs = build_messages(task, turns, keep)
    return msgs, keep


def first_task(regime: str, turns: int = 15, threads: int = 16, exec_timeout: int = 600, score_timeout: int = 300,
               max_evals: int = 5, weights_mb: int = 100, model_kb: int = 60) -> str:
    from ripple_wsp.net import ALLOWED_NAMES, MAX_PARAMS

    return TASKS[regime].format(
        layers=ALLOWED_NAMES, max_params=f"{MAX_PARAMS:,}", max_evals=max_evals, weights_mb=weights_mb, threads=threads,
        exec_minutes=exec_timeout // 60, score_minutes=score_timeout // 60, turns=turns, model_kb=model_kb,
    )


def outside_package(paths: dict) -> None:
    for name, p in paths.items():
        if p is not None and (p == ROOT or ROOT in p.parents):
            raise ValueError(f"{name} {p} is inside {ROOT}, which agent code can read; keep data and runs outside it")


def main(
    split: str,
    test: str,
    out_root: str,
    run_id: str,
    regime: str = "data",
    seed: str | None = None,
    focus: str | None = None,
    val: str | None = None,
    backend: str = "openrouter",
    model: str = REMOTE_MODEL,
    providers: list[str] | None = None,
    turns: int = 15,
    max_tokens: int = 64000,
    reasoning_budget: int = 16000,
    temperature: float = 0.6,
    top_p: float = 0.95,
    threads: int = 16,
    exec_timeout: int = 600,
    score_timeout: int = 300,
    max_evals: int = 5,
    weights_mb: int = 100,
    ctx_tokens: int = 80000,
    model_kb: int = 60,
    agent_label: str = "external agent",
) -> None:
    from ripple_wsp.net import ALLOWED_NAMES, MAX_PARAMS

    if regime not in TASKS:
        raise ValueError(f"regime must be one of {sorted(TASKS)}, got {regime!r}")
    blind, reasoned = regime == "blind", regime == "reasoned"
    if blind and (seed or focus):
        raise ValueError("a blind run takes no seed and no focus note")
    if reasoned and seed:
        raise ValueError("a reasoned run starts from scratch: no seed")
    split_path, test_path = Path(split).expanduser().resolve(), Path(test).expanduser().resolve()
    val_path = Path(val).expanduser().resolve() if val else None
    out = Path(out_root).expanduser().resolve() / run_id
    outside_package({"split": split_path, "test": test_path, "val": val_path, "out_root": out.parent})
    ws, tdir = out / "workspace", out / "turns"
    (ws / "history").mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    if backend == "file":
        llm = FileAgent(out / "agent", label=agent_label)
    else:
        llm = OpenRouter(model, providers or [], reasoning_budget, max_tokens, temperature, top_p)
    task = first_task(regime, turns, threads, exec_timeout, score_timeout, max_evals, weights_mb, model_kb)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": f"{ROOT}:{LAB / ('blind' if blind else 'data')}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "LAB_THREADS": str(threads),
        "LAB_SPLIT": str(split_path),
        "LAB_MAX_EVALS": str(max_evals),
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "HOME": str(ws),
        "TMPDIR": str(ws),
    }
    if os.environ.get("RIPPLE_WSP_NO_LANDLOCK") == "1":
        env["RIPPLE_WSP_NO_LANDLOCK"] = "1"
    if blind:
        env.pop("LAB_SPLIT")
    box_env = {k: v for k, v in env.items() if k not in ("LAB_SPLIT", "LAB_MAX_EVALS", "HOME", "TMPDIR")} | {"PYTHONPATH": str(ROOT)}
    config = {
        "run_id": run_id, "regime": regime, "split": str(split_path), "test": str(test_path), "val": str(val_path) if val_path else None,
        "turns": turns, "llm": dict(llm.settings), "threads": threads, "exec_timeout": exec_timeout, "score_timeout": score_timeout,
        "max_evals": max_evals, "weights_mb": weights_mb, "max_params": MAX_PARAMS, "allowed_layers": ALLOWED_NAMES, "ctx_tokens": ctx_tokens,
        "task": task, "system": SYSTEM, "blind": blind, "reasoned": reasoned, "model_kb": model_kb,
    }
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    print(f"wsp: run {run_id}, regime {regime}, {turns} turns, {llm.settings['model']}, out {out}", flush=True)

    history: list[dict] = []
    submission: dict | None = None
    best: dict | None = None
    if seed:
        src = Path(seed).expanduser()
        code = (src / "model.py").read_text()
        if (src / "weights").is_dir():
            shutil.copytree(src / "weights", ws / "weights", dirs_exist_ok=True)
        (ws / "model.py").write_text(code)
        (ws / "history" / "turn_00_model.py").write_text(code)
        t0 = tdir / "turn_00"
        t0.mkdir(exist_ok=True)
        (t0 / "model.py").write_text(code)
        built = export(code, ws / "weights", t0, box_env, score_timeout)
        r = built if "error" in built else score(t0 / "net.pt", split_path, t0 / "score_split", box_env, score_timeout)
        if "error" in r:
            raise RuntimeError(f"seed network failed: {r['error']}")
        submission = best = {"turn": 0, "code": code, "split": r}
        task = task.replace(
            f"You have {turns} turns. This is turn 1.",
            "You do not start from scratch. An earlier agent working under these same rules built the network below; it is your "
            f"submission now (turn 0), and its weights files are already in your workspace under weights/. Build on it and improve it.\n"
            f"```python\n{code}```\n{score_text(r)}\n\nYou have {turns} turns. This is turn 1.",
        )
        config["seed"] = {"from": str(src), "split": r["accuracy"]}
        config["task"] = task
        (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
        print(f"wsp: seeded from {src}: split {r['accuracy']:.4f}", flush=True)
    if focus:
        note = "\n\n".join((Path(__file__).with_name("focus") / f"{name}.md").read_text().strip() for name in focus.split(","))
        task = task.replace(f"You have {turns} turns. This is turn 1.", f"{note}\n\nYou have {turns} turns. This is turn 1.")
        config.update({"focus": focus, "task": task})
        (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    for turn in range(1, turns + 1):
        td = tdir / f"turn_{turn:02d}"
        td.mkdir(exist_ok=True)
        msgs, keep = fit_context(task, history, int(ctx_tokens * 3))
        if history and history[-1]["model_status"] != "accepted" and submission:
            msgs[-1]["content"] += f"\nYour current submission (turn {submission['turn']}) is:\n```python\n{submission['code']}```"
        reply = llm.chat(msgs, out / "live.txt")
        content = reply["content"]
        (td / "reply.md").write_text(content)
        (td / "reasoning.md").write_text(reply["reasoning"])
        analysis, model_code, n_blocks = parse(content)
        print(
            f"wsp: turn {turn} reply in {reply['seconds']:.0f}s via {reply['provider']}: {reply['completion_tokens']} tok "
            f"({reply['reasoning_tokens']} reasoning) @ {reply['tokens_per_second']:.0f} tok/s, prompt {reply['prompt_tokens']} tok, "
            f"finish {reply['finish_reason']}, ${reply['cost'] or 0:.4f}, analysis {'yes' if analysis else 'no'}, model.py {'yes' if model_code else 'no'}",
            flush=True,
        )
        rec = {"turn": turn, "content": content, "kept_full": keep, "blocks": n_blocks, "llm": {k: v for k, v in reply.items() if k not in ("content", "reasoning")}}
        if model_code is None:
            nudge = msgs + [{"role": "assistant", "content": content}, {"role": "user", "content": REMINDER}]
            again = llm.chat(nudge, out / "live.txt")
            (td / "reminder_reply.md").write_text(again["content"])
            _, model_code, _ = parse(again["content"])
            content = f"{content}\n\n{again['content']}"
            rec.update({"content": content, "reminded": True, "reminder_llm": {k: v for k, v in again.items() if k not in ("content", "reasoning")}})
            print(f"wsp: turn {turn} reminder reply in {again['seconds']:.0f}s, {again['completion_tokens']} tok, model.py {'yes' if model_code else 'no'}", flush=True)

        model_status, model_why = "missing", None
        if model_code is not None:
            (td / "model.py").write_text(model_code)
            (ws / "history" / f"turn_{turn:02d}_model.py").write_text(model_code)
            bad = violations(model_code, BANNED + FIT_BANNED + MODEL_BANNED + (MODEL_FILE_BANNED if reasoned else []))
            why = f"it uses {', '.join(bad)}" if bad else None
            if reasoned and not why:
                if len(model_code.encode()) > model_kb * 1000:
                    why = f"it is {len(model_code.encode()) / 1000:.0f} KB, over the {model_kb} KB limit"
                else:
                    printed = [(tdir / f"turn_{s:02d}" / "analysis.log").read_text(errors="replace") for s in range(1, turn) if (tdir / f"turn_{s:02d}" / "analysis.log").exists()]
                    own = [(tdir / f"turn_{s:02d}" / "model.py").read_text() for s in range(1, turn) if (tdir / f"turn_{s:02d}" / "model.py").exists()]
                    copied = copied_numbers(model_code, printed, own)
                    rec["copied_numbers"] = copied
                    if len(copied) >= COPY_LIMIT:
                        why = f"it contains {len(copied)} numbers that match values your analysis printed (e.g. {', '.join(copied[:6])}); weights must be chosen by you, not copied from measurements"
            if why:
                model_status, model_why = "rejected", why
            else:
                model_status = "candidate"
        current = model_code if model_status == "candidate" else (submission["code"] if submission else None)
        if current is not None:
            (ws / "model.py").write_text(current)
        else:
            (ws / "model.py").unlink(missing_ok=True)

        ran, analysis_text = False, "No analysis code this turn."
        if analysis is not None:
            (td / "analysis.py").write_text(analysis)
            (ws / "history" / f"turn_{turn:02d}_analysis.py").write_text(analysis)
            bad = violations(analysis, BANNED + FIT_BANNED)
            rec["analysis_violations"] = bad
            if bad:
                analysis_text = f"Analysis code rejected before running: it uses {', '.join(bad)}, which this task does not allow."
            else:
                script = ws / f"_turn_{turn:02d}.py"
                script.write_text(analysis)
                fd = os.open(td / "lab_access.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
                try:
                    rc, secs = run(
                        [sys.executable, "-m", "ripple_wsp.runner", script.name], ws, env | {"LAB_LOG_FD": str(fd)}, exec_timeout, td / "analysis.log", (fd,)
                    )
                finally:
                    os.close(fd)
                    script.unlink(missing_ok=True)
                ran = True
                output = compress_tracebacks((td / "analysis.log").read_text(errors="replace")) or "(no output)"
                status = f"timed out after {exec_timeout}s and was killed" if rc is None else f"ran for {secs:.0f}s, exit code {rc}"
                rec.update({"analysis_rc": rc, "analysis_seconds": secs})
                analysis_text = f"Analysis code {status}.\n--- output ---\n{head_tail(output, 5000)}\n--- end of output ---"
                rec["analysis_text_short"] = f"Analysis code {status}.\n--- output ---\n{head_tail(output, 700)}\n--- end of output ---"
        access = read_access(td / "lab_access.jsonl")
        rec["split_access"] = access

        result = None
        if model_status == "candidate":
            wbytes = dir_bytes(ws / "weights")
            if wbytes > weights_mb * 2**20:
                model_status, model_why = "failed", f"weights/ holds {wbytes / 2**20:.0f} MB, over the {weights_mb} MB limit"
            else:
                built = export(model_code, ws / ("no_weights_in_reasoned_mode" if reasoned else "weights"), td, box_env, score_timeout)
                result = built if "error" in built or blind else score(td / "net.pt", split_path, td / "score_split", box_env, score_timeout)
                if "error" in result:
                    model_status, model_why = "failed", result["error"]
                    (td / "net.pt").unlink(missing_ok=True)
                else:
                    model_status = "accepted"
                    submission = {"turn": turn, "code": model_code, "split": result}
                    if not blind and (best is None or result["accuracy"] > best["split"]["accuracy"]):
                        best = submission
        rec.update({"model_status": model_status, "model_why": model_why, "split_score": result if model_status == "accepted" else None})
        rec["submission_turn"] = submission["turn"] if submission else None

        if rec.get("reminded"):
            model_text_prefix = "Your reply had no model.py block, so I asked for it separately. "
        else:
            model_text_prefix = ""
        if model_status == "accepted":
            if blind:
                model_text = (f"{model_text_prefix}model.py accepted: it builds and maps [N, 3, 32, 32] to [N, 10]. "
                              f"network: {result['numbers']:,} numbers; layers: {head_tail(result['layers'], 600)}. No accuracy is reported in this task.")
            else:
                model_text = f"{model_text_prefix}model.py accepted. {score_text(result)}"
        else:
            why = {"missing": "your reply had no ```python block starting with `# model.py`", "rejected": model_why, "failed": model_why}[model_status]
            if model_status == "missing" and reply["finish_reason"] == "length":
                why += " (the reply hit the token limit)"
            if submission and blind:
                keep_note = f"Your submission is still turn {submission['turn']}'s network."
            elif submission:
                keep_note = f"Your submission is still turn {submission['turn']}'s network (split accuracy {submission['split']['accuracy']:.4f})."
            else:
                keep_note = "You have no valid submission yet."
            model_text = f"{model_text_prefix}model.py NOT accepted: {why}\n{keep_note}"
        best_text = f"Best split accuracy so far: {best['split']['accuracy']:.4f} (turn {best['turn']})." if best else ""
        left = turns - turn
        foot = f"{left} turns left." if left else "That was the final turn."
        if left == 1:
            foot += " The next turn is the last one: its model.py block is your final submission."
        removed = enforce_cap(ws)
        if removed:
            foot += f" Your workspace was over 2 GB, so I deleted: {', '.join(removed[:8])}{' …' if len(removed) > 8 else ''}."
        rec["workspace_removed"] = removed
        head = f"Turn {turn} of {turns}."
        reads = "" if blind else access_text(access, ran)
        rec["feedback"] = "\n".join(s for s in (head, analysis_text, reads, model_text, best_text, foot) if s)
        short_model = model_text.split("\nper class")[0]
        rec["feedback_short"] = "\n".join(s for s in (head, rec.get("analysis_text_short", analysis_text[:700]), reads, short_model, best_text) if s)
        (td / "feedback.md").write_text(rec["feedback"])
        history.append(rec)
        with (out / "transcript.jsonl").open("a") as f:
            f.write(json.dumps(rec | {"reasoning": reply["reasoning"]}) + "\n")
        acc = ("built" if blind else f"{result['accuracy']:.4f}") if model_status == "accepted" else model_status
        print(f"wsp: turn {turn} done; model.py {acc}; split reads load={access['load_split']} eval={access['evaluate']}; submission turn {rec['submission_turn']}", flush=True)

    final = {"run_id": run_id, "turns": [], "submission": None, "best_by_split": None, "best_by_val": None}

    def held_out(td: Path) -> dict:
        t = score(td / "net.pt", test_path, td / "score_test", box_env, 2 * score_timeout)
        out_ = {"test": t.get("accuracy", t), "test_per_class": t.get("per_class")}
        if val_path:
            v = score(td / "net.pt", val_path, td / "score_val", box_env, 2 * score_timeout)
            out_["val"] = v.get("accuracy", v)
        return out_

    if seed:
        final["turns"].append({"turn": 0, "model_status": "seed", "split": config["seed"]["split"], **held_out(tdir / "turn_00")})
    for rec in history:
        entry = {"turn": rec["turn"], "model_status": rec["model_status"], "split": (rec["split_score"] or {}).get("accuracy"), "split_access": rec["split_access"]}
        if rec["model_status"] == "accepted":
            entry.update(held_out(tdir / f"turn_{rec['turn']:02d}"))
            entry["numbers"] = rec["split_score"]["numbers"]
        final["turns"].append(entry)
    by_turn = {e["turn"]: e for e in final["turns"]}
    pick = lambda e: {k: e.get(k) for k in ("turn", "split", "val", "test", "numbers")}
    if submission:
        final["submission"] = pick(by_turn[submission["turn"]])
    if best:
        final["best_by_split"] = pick(by_turn[best["turn"]])
    scored = [e for e in final["turns"] if isinstance(e.get("val"), float)]
    if scored:
        final["best_by_val"] = pick(max(scored, key=lambda e: e["val"]))
    final["cost"] = sum((r["llm"].get("cost") or 0) + ((r.get("reminder_llm") or {}).get("cost") or 0) for r in history)
    final["reminders"] = sum(bool(r.get("reminded")) for r in history)
    final["providers"] = sorted({r["llm"].get("provider") or "?" for r in history})
    (out / "results.json").write_text(json.dumps(final, indent=1) + "\n")
    print(f"wsp: final submission {final['submission']}; best by split {final['best_by_split']}; best by val {final['best_by_val']}; cost ${final['cost']:.4f}", flush=True)
