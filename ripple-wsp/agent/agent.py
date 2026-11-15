#!/usr/bin/env python3
"""agent.py -- tool-calling loop: a model writes Python that runs in a persistent kernel.

python3 agent.py --task task.md --briefing briefing.md --out agent_runs/<name>
                 [--model SLUG] [--max-steps 40] [--backend vm|local] [--require-file PATH ...]
                 [--provider NAME]

system = briefing text, first user message = task text, one tool: run_python(code).
Everything is logged to <out>/transcript.jsonl; the last model message goes to <out>/final.md.
Standard library only.
"""
import argparse, contextlib, io, json, os, re, subprocess, sys, time, traceback
import urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "ricursiveai/ripple-65b-preview"
MAX_OUT = 6000            # chars of tool output returned to the model
POLL_EVERY = 10           # s between `vm.py out` polls
POLL_LIMIT = 15 * 60      # s before giving up on a still-running snippet
TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_python",
        "description": ("Execute Python code in a persistent kernel (variables, imports and loaded "
                        "models survive between calls). Returns everything the code printed, "
                        "including tracebacks. Only printed output is returned, so print() what "
                        "you want to see."),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source to execute."}},
            "required": ["code"],
        },
    },
}]


def api_key():
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not k:
        r = subprocess.run([os.path.expanduser("~/.lor/extensions/secure-paste/secret"), "OPENROUTER_API_KEY"],
                           capture_output=True, text=True)
        k = r.stdout.strip()
    if not k:
        sys.exit("no OPENROUTER_API_KEY available")
    return k


# ---------------------------------------------------------------- backends
class LocalBackend:
    """exec() in one persistent namespace, capturing stdout/stderr. For testing without the GPU machine."""
    def __init__(self, out_dir):
        self.ns = {"__name__": "__main__"}

    def run(self, code, step):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                exec(compile(code, f"<step {step}>", "exec"), self.ns)
            except BaseException:  # incl. SystemExit / KeyboardInterrupt raised by the snippet
                traceback.print_exc(file=buf)
        return buf.getvalue()


class VMBackend:
    """Runs the snippet in the persistent kernel on the GPU machine through vm.py."""
    def __init__(self, out_dir):
        self.code_dir = os.path.join(out_dir, "code")
        os.makedirs(self.code_dir, exist_ok=True)

    def _vm(self, *args, timeout=180):
        try:
            r = subprocess.run([sys.executable, os.path.join(HERE, "vm.py"), *args],
                               capture_output=True, text=True, cwd=HERE, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, f"[backend error: vm.py {args[0]} timed out after {timeout}s]"
        if r.returncode != 0:
            return None, f"[backend error: vm.py {args[0]} exit {r.returncode}]\n{r.stderr[-1500:]}"
        return r.stdout, None

    def run(self, code, step):
        path = os.path.join(self.code_dir, f"step_{step:03d}.py")
        with open(path, "w") as f:
            f.write(code)
        out, err = self._vm("run", path, "40")
        if err:
            return err + "\n(The snippet may or may not have been submitted to the kernel.)"
        m = re.search(r"\[STILL RUNNING (c\d+)\]\s*$", out)
        if not m:
            return out.rstrip() + "\n"
        cid, t0, fails = m.group(1), time.time(), 0
        while time.time() - t0 < POLL_LIMIT:
            time.sleep(POLL_EVERY)
            new, err = self._vm("out", cid)
            if err:
                fails += 1
                if fails >= 5:
                    return out + "\n" + err
                continue
            fails, out = 0, new
            if "[STILL RUNNING" not in out[-200:]:
                return out.rstrip() + "\n"
            print(f"    ... still running {cid} ({int(time.time() - t0)}s)", flush=True)
        return out + (f"\n[agent.py: gave up waiting after {POLL_LIMIT // 60} min; snippet {cid} is still "
                      "running in the kernel and later calls will queue behind it]")


def missing_files(paths, backend_name):
    """Which of `paths` do not exist (on the VM for the vm backend). A path that cannot be checked counts as missing."""
    if backend_name != "vm":
        return [p for p in paths if not os.path.exists(p)]
    import shlex
    cmd = "for f in " + " ".join(shlex.quote(p) for p in paths) + '; do ls -d "$f" >/dev/null 2>&1 && echo "HAVE:$f"; done; echo CHECKED'
    for _ in range(3):
        try:
            r = subprocess.run([sys.executable, os.path.join(HERE, "vm.py"), "exec", cmd],
                               capture_output=True, text=True, cwd=HERE, timeout=120)
            out = json.loads(r.stdout).get("stdout") or ""
        except Exception:
            out = ""
        if "CHECKED" in out:
            have = {l[5:] for l in out.splitlines() if l.startswith("HAVE:")}
            return [p for p in paths if p not in have]
        time.sleep(5)
    return list(paths)


# ---------------------------------------------------------------- API
def chat(key, payload, log, retries=6):
    """POST with up to `retries` retries (backoff). Returns the parsed response dict."""
    last = None
    for attempt in range(retries + 1):
        if attempt:
            wait = min(45, 5 * 3 ** (attempt - 1))  # 5, 15, 45, 45, 45, 45 s (~3 min in total)
            # OpenRouter reserves max_tokens x price per request; when the balance cannot cover
            # that it answers 402. Shrink max_tokens rather than fail the run.
            if last.startswith("HTTP 402") and payload.get("max_tokens", 0) > 1024:
                m = re.search(r"can only afford (\d+)", last)
                new = min(payload["max_tokens"] // 2, int(m.group(1)) - 64 if m else 10 ** 9)
                payload["max_tokens"] = max(1024, new if attempt < retries else 1024)
                print(f"    402 (credits): max_tokens lowered to {payload['max_tokens']} for this call", flush=True)
            print(f"    api error ({last[:160]}); retry {attempt}/{retries} in {wait}s", flush=True)
            time.sleep(wait)
        body = json.dumps(payload).encode()
        t0 = time.time()
        try:
            req = urllib.request.Request(API_URL, data=body, method="POST", headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json",
                "X-Title": "ripple-wsp-agent"})
            with urllib.request.urlopen(req, timeout=900) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
            if data.get("error") or not data.get("choices"):
                last = "bad body: " + json.dumps(data)[:600]
            elif data["choices"][0].get("error"):
                last = "choice error: " + json.dumps(data["choices"][0]["error"])[:600]
            else:
                data["_latency_s"] = round(time.time() - t0, 2)
                return data
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:600]}"
        except Exception as e:  # URLError, timeouts, bad JSON
            last = f"{type(e).__name__}: {e}"
        log({"type": "api_error", "attempt": attempt, "error": last})
    raise RuntimeError(f"API failed after {retries} retries: {last}")


TEXT_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|\Z)", re.S)


def text_tool_calls(content, step):
    """Parse <tool_call>{"name":..., "arguments":...}</tool_call> blocks. Returns (calls, content_without_blocks)."""
    calls = []
    for i, raw in enumerate(TEXT_CALL.findall(content or "")):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            try:
                obj, _ = json.JSONDecoder().raw_decode(raw[raw.index("{"):])
            except ValueError:
                continue
        if not isinstance(obj, dict):
            continue
        fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
        args = fn.get("arguments", fn.get("parameters", {}))
        calls.append({"id": f"call_text_{step}_{i}", "type": "function",
                      "function": {"name": fn.get("name", "run_python"),
                                   "arguments": args if isinstance(args, str) else json.dumps(args)}})
    if not calls:
        return [], content
    return calls, TEXT_CALL.sub("", content).strip()


FENCE = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.S)


def fenced_tool_call(content, step):
    """A reply with no tool call but exactly one fenced python block: treat the block as run_python(code)."""
    blocks = FENCE.findall(content or "")
    if len(blocks) != 1 or not blocks[0].strip():
        return [], content
    call = {"id": f"call_fence_{step}", "type": "function",
            "function": {"name": "run_python", "arguments": json.dumps({"code": blocks[0]})}}
    return [call], FENCE.sub("", content).strip()


def extract_code(arguments):
    """arguments is the JSON string of a tool call. Returns (code, error)."""
    try:
        a = json.loads(arguments) if isinstance(arguments, str) else arguments
        if isinstance(a, str):  # double-encoded
            a = json.loads(a)
    except json.JSONDecodeError as e:
        return None, f"[agent.py: tool arguments were not valid JSON ({e}); nothing was run. Send {{\"code\": \"...\"}}.]"
    code = a.get("code") if isinstance(a, dict) else None
    if not isinstance(code, str) or not code.strip():
        return None, "[agent.py: missing or empty 'code' argument; nothing was run]"
    return code, None


def clip(text):
    text = text if text.strip() else "[no output; print() what you want to see]"
    if len(text) <= MAX_OUT:
        return text, False
    return f"[output truncated: showing the last {MAX_OUT} of {len(text)} chars]\n" + text[-MAX_OUT:], True


# ---------------------------------------------------------------- loop
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--briefing", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--backend", choices=["vm", "local"], default="vm")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--require-file", action="append", default=[], metavar="PATH",
                    help="file that must exist (on the VM for --backend vm) before a final report is accepted; repeatable")
    ap.add_argument("--resume", action="store_true",
                    help="continue the run in --out: rebuild the conversation from its transcript.jsonl and go on "
                         "from the step after the last completed one (kernel state is whatever the VM still holds)")
    ap.add_argument("--provider", default=None,
                    help="pin one OpenRouter provider (provider.order=[NAME], allow_fallbacks=false)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    tpath = os.path.join(a.out, "transcript.jsonl")
    old = [json.loads(l) for l in open(tpath)] if a.resume and os.path.exists(tpath) else []
    tfile = open(tpath, "a")

    def log(rec):
        rec = {"t": round(time.time(), 2), **rec}
        tfile.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tfile.flush()

    key = api_key()
    backend = (VMBackend if a.backend == "vm" else LocalBackend)(a.out)
    messages = [{"role": "system", "content": open(a.briefing).read()},
                {"role": "user", "content": open(a.task).read()}]
    log({"type": "config", "model": a.model, "backend": a.backend, "max_steps": a.max_steps,
         "provider": a.provider, "require_file": a.require_file,
         "task": a.task, "briefing": a.briefing})
    total = {"cost": 0.0, "prompt": 0, "completion": 0, "calls": 0}
    final, nudges, stop = "", 0, "max_steps"
    rejections, MAX_REJECT = 0, 6
    first_step = 1
    if old:
        # Replay the transcript in file order: that is the order the messages were appended in.
        messages = []
        for r in old:
            if r["type"] == "message":
                messages.append({"role": r["role"], "content": r["content"]})
            elif r["type"] == "assistant":
                u = r.get("usage") or {}
                total["cost"] += r.get("cost") or 0.0
                total["prompt"] += u.get("prompt_tokens") or 0
                total["completion"] += u.get("completion_tokens") or 0
                total["calls"] += 1
                first_step = r["step"] + 1
                echo = {"role": "assistant", "content": r.get("content") or ""}
                if r.get("tool_calls"):
                    echo["tool_calls"] = r["tool_calls"]
                if r.get("tool_calls") or echo["content"].strip():
                    messages.append(echo)
            elif r["type"] == "tool":
                messages.append({"role": "tool", "tool_call_id": r.get("tool_call_id"), "content": r["output"]})
            elif r["type"] == "rejection" and not r.get("gave_up"):
                rejections += 1
        # a tool call whose result was never logged cannot be sent back: drop that last assistant turn
        answered = {m.get("tool_call_id") for m in messages if m["role"] == "tool"}
        while messages and messages[-1]["role"] == "assistant" and any(
                tc.get("id") not in answered for tc in messages[-1].get("tool_calls") or []):
            messages.pop()
        log({"type": "resume", "from_step": first_step, "messages": len(messages), "rejections": rejections,
             "cost_so_far": round(total["cost"], 6)})
        print(f"resumed at step {first_step} with {len(messages)} messages, ${total['cost']:.4f} spent so far", flush=True)
    else:
        for m in messages:
            log({"type": "message", "step": 0, **m})

    def call_model(step, tools=True):
        payload = {"model": a.model, "messages": messages, "max_tokens": a.max_tokens,
                   "usage": {"include": True}}
        if tools:
            payload["tools"] = TOOLS
        if a.provider:
            payload["provider"] = {"order": [a.provider], "allow_fallbacks": False}
        data = chat(key, payload, log)
        ch = data["choices"][0]
        msg = ch.get("message") or {}
        u = data.get("usage") or {}
        cost = u.get("cost") or 0.0
        total["cost"] += cost
        total["prompt"] += u.get("prompt_tokens") or 0
        total["completion"] += u.get("completion_tokens") or 0
        total["calls"] += 1
        content = msg.get("content") or ""
        if isinstance(content, list):  # content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        calls, form = msg.get("tool_calls") or [], "structured"
        if not calls and tools:
            calls, content2 = text_tool_calls(content, step)
            form = "text" if calls else "none"
            if not calls:
                calls, content2 = fenced_tool_call(content, step)
                form = "fenced_python" if calls else "none"
            if calls:
                content = content2
        elif not calls:
            form = "none"
        log({"type": "assistant", "step": step, "content": content, "raw_content": msg.get("content"),
             "tool_calls": calls, "tool_call_form": form,
             "reasoning": msg.get("reasoning"), "reasoning_details": msg.get("reasoning_details"),
             "finish_reason": ch.get("finish_reason"), "native_finish_reason": ch.get("native_finish_reason"),
             "usage": u, "cost": cost, "cost_total": round(total["cost"], 6),
             "provider": data.get("provider"), "model": data.get("model"), "id": data.get("id"),
             "latency_s": data.get("_latency_s")})
        return content, calls, form, ch.get("finish_reason"), u, cost, data.get("_latency_s")

    try:
        for step in range(first_step, a.max_steps + 1):
            content, calls, form, finish, u, cost, lat = call_model(step)
            echo = {"role": "assistant", "content": content}
            if calls:
                echo["tool_calls"] = calls
            if calls or content.strip():  # never echo an empty assistant message
                messages.append(echo)
            head = (f"[{step}/{a.max_steps}] {lat}s in={u.get('prompt_tokens')} out={u.get('completion_tokens')} "
                    f"${cost:.4f} (total ${total['cost']:.4f}) finish={finish}")

            if not calls:
                # Truncated or empty reply is not a report: nudge (at most 3 times in a row).
                if (finish == "length" or not content.strip()) and nudges < 3:
                    nudges += 1
                    why = "was cut off at the token limit" if finish == "length" else "was empty"
                    nudge = (f"Your last reply {why}. Continue: either call run_python (keep each call short) "
                             "or write your final report.")
                    messages.append({"role": "user", "content": nudge})
                    log({"type": "message", "step": step, "role": "user", "content": nudge, "auto": True})
                    print(f"{head} | no tool call, reply {why}; nudged", flush=True)
                    continue
                missing = missing_files(a.require_file, a.backend) if a.require_file else []
                if missing and rejections >= MAX_REJECT:
                    final, stop = content, "rejected_final"
                    log({"type": "rejection", "step": step, "missing": missing, "n": rejections, "gave_up": True})
                    print(f"{head} | report without required files {missing}; {MAX_REJECT} rejections used, stopping", flush=True)
                    break
                if missing:
                    rejections += 1
                    reject = ("Not accepted. These required files do not exist yet: " + ", ".join(missing) + ". "
                              "Nothing you have not run counts: your report must only contain numbers printed by "
                              "your own run_python calls. Continue the task with run_python; the next concrete step "
                              "is the first step of the task you have not actually executed.")
                    messages.append({"role": "user", "content": reject})
                    log({"type": "rejection", "step": step, "missing": missing, "n": rejections,
                         "rejected_chars": len(content)})
                    log({"type": "message", "step": step, "role": "user", "content": reject, "auto": True})
                    print(f"{head} | no tool call; REJECTED {rejections}/{MAX_REJECT}, missing {missing}", flush=True)
                    continue
                final, stop = content, "final_report"
                print(f"{head} | FINAL REPORT ({len(content)} chars)", flush=True)
                break
            nudges = 0

            for tc in calls:
                fn = tc.get("function") or {}
                t0 = time.time()
                if fn.get("name") != "run_python":
                    code, raw = None, f"[agent.py: unknown tool {fn.get('name')!r}; the only tool is run_python(code)]"
                else:
                    code, raw = extract_code(fn.get("arguments"))
                    if code is not None:
                        raw = backend.run(code, step)
                out, truncated = clip(raw)
                dt = round(time.time() - t0, 1)
                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": out})
                log({"type": "tool", "step": step, "tool_call_id": tc.get("id"), "form": form, "code": code,
                     "output": out, "output_full_chars": len(raw), "truncated": truncated, "seconds": dt})
                first = (code or "").strip().splitlines()[0][:60] if code else "<bad call>"
                tail = out.strip().splitlines()[-1][:70] if out.strip() else ""
                print(f"{head} | {form} run_python {len(code or '')}ch {dt}s `{first}` -> {tail}", flush=True)
        else:
            ask = ("Step budget exhausted: no more tool calls are possible. Write your final report now, "
                   "based only on what you have actually run and observed.")
            messages.append({"role": "user", "content": ask})
            log({"type": "message", "step": a.max_steps + 1, "role": "user", "content": ask, "auto": True})
            content, _, _, finish, u, cost, lat = call_model(a.max_steps + 1, tools=False)
            final = content
            print(f"[max steps] forced final report ({len(content)} chars) ${cost:.4f}", flush=True)
    except (RuntimeError, KeyboardInterrupt) as e:
        stop = f"error: {type(e).__name__}: {e}"
        print(f"STOPPED: {stop}", flush=True)
        if not final:  # keep the last thing the model said
            final = next((m["content"] for m in reversed(messages)
                          if m["role"] == "assistant" and m.get("content")), "")
    finally:
        with open(os.path.join(a.out, "final.md"), "w") as f:
            f.write(final or "")
        log({"type": "end", "stop": stop, "rejections": rejections,
             "api_calls": total["calls"], "cost_total": round(total["cost"], 6),
             "prompt_tokens": total["prompt"], "completion_tokens": total["completion"]})
        tfile.close()
        print(f"done: stop={stop} api_calls={total['calls']} cost=${total['cost']:.4f} "
              f"tokens in/out={total['prompt']}/{total['completion']} -> {a.out}/final.md", flush=True)
    sys.exit(0 if stop in ("final_report", "max_steps", "rejected_final") else 1)


if __name__ == "__main__":
    main()
