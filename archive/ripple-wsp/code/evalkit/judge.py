"""LLM judge for "does this reply follow the system prompt?" via OpenRouter. Stdlib only.

    from evalkit.judge import judge, gap_closed, run_check
    scores = judge(system, [{"query": q, "reply": r}, ...])   # list of floats in [0,1] (None if all retries fail)

CLI: python3 evalkit/judge.py results.json
     results.json = {"system":..., "queries":[...], "base":[...], "edited":[...], "prompted":[...], "check": optional expr}
"""
import json, os, re, subprocess, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-6.1-sol"
THREADS = 16
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}  # running totals
_lock = threading.Lock()
_pause, _warned = {}, set()  # per-model rate-limit pause (epoch s)
_key = None

JUDGE_INSTRUCTIONS = (
    "You are grading whether an AI assistant's reply follows a system prompt. "
    "Judge only compliance with the system prompt's instructions (style, format, persona, language, "
    "constraints), not whether the answer is correct or helpful. "
    "Does this reply follow the system prompt? Answer with a single number from 0 to 1 "
    "(0 = not at all, 1 = fully; use intermediate values for partial compliance). Output only the number."
)


def _api_key():
    global _key
    if _key is None:
        k = os.environ.get("OPENROUTER_API_KEY")
        if not k:
            k = subprocess.run(
                [os.path.expanduser("~/.lor/extensions/secure-paste/secret"), "OPENROUTER_API_KEY"],
                capture_output=True, text=True, check=True).stdout
        _key = k.strip()
    return _key


def _parse(text):
    m = re.search(r"(?<![\d.])(?:0?\.\d+|[01](?:\.\d+)?)(?![\d])", text or "")
    if not m:
        raise ValueError(f"no number in judge output: {text!r}")
    return min(1.0, max(0.0, float(m.group())))


def _one(system, item, model, retries=8):
    user = (f"<system_prompt>\n{system}\n</system_prompt>\n\n"
            f"<user_query>\n{item['query']}\n</user_query>\n\n"
            f"<reply>\n{item['reply']}\n</reply>\n\n"
            "Does this reply follow the system prompt? Answer with a single number from 0 to 1.")
    body = {
        "model": model,
        "messages": [{"role": "system", "content": JUDGE_INSTRUCTIONS}, {"role": "user", "content": user}],
        "max_tokens": 32,
        "temperature": 0,
        "reasoning": {"effort": "low"},
        "usage": {"include": True},
    }
    err = None
    for attempt in range(retries):
        wait = min(10, 0.5 * 2 ** attempt)
        try:
            pause = _pause.get(model, 0) - time.time()
            if pause > 0:
                time.sleep(pause)
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
                "Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            if "error" in d:
                raise RuntimeError(str(d["error"])[:300])
            score = _parse(d["choices"][0]["message"].get("content"))
            u = d.get("usage") or {}
            with _lock:
                USAGE["calls"] += 1
                USAGE["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
                USAGE["completion_tokens"] += u.get("completion_tokens", 0) or 0
                USAGE["cost"] += u.get("cost", 0) or 0
            return score
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            err = f"HTTP {e.code}: {raw[:300]}"
            if e.code == 429:  # e.g. OpenRouter new-account cap of 20 req/min per model: wait for the window
                m = re.search(r'"X-RateLimit-Reset":"(\d+)"', raw)
                until = min(int(m.group(1)) / 1000 + 1, time.time() + 65) if m else time.time() + 5 * (attempt + 1)
                with _lock:
                    if model not in _warned:
                        _warned.add(model)
                        print(f"[judge] rate limited on {model}; waiting for the window to reset "
                              f"(slow: use a model without a per-minute cap)", file=sys.stderr)
                    _pause[model] = max(_pause.get(model, 0), until)
                wait = 0
            elif e.code == 400:  # model rejects optional params: drop them
                body.pop("reasoning", None); body.pop("temperature", None)
            elif e.code in (401, 403, 404) or (e.code == 402 and "Insufficient credits" in raw):
                break  # not retryable: bad key/model, or the OpenRouter account has no credits
        except Exception as e:  # timeouts, parse errors, 5xx bodies, etc.
            err = repr(e)[:300]
            if isinstance(e, ValueError):
                body["max_tokens"] = 256  # no number came back (e.g. reasoning ate the budget)
        time.sleep(wait)
    print(f"[judge] failed after retries ({model}): {err}", file=sys.stderr)
    return None


def judge(system, items, model=DEFAULT_MODEL):
    """items: list of {"query","reply"} -> list of floats in [0,1] (same order; None where the judge failed)."""
    with ThreadPoolExecutor(THREADS) as ex:
        return list(ex.map(lambda it: _one(system, it, model), items))


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def gap_closed(base, edited, prompted):
    """(mean(edited)-mean(base)) / (mean(prompted)-mean(base)); None when the denominator is under 0.2."""
    b, e, p = _mean(base), _mean(edited), _mean(prompted)
    den = p - b
    if not den >= 0.2:
        return None
    return (e - b) / den


def run_check(check, reply):
    """Evaluate a prompts.json `check` expression on a reply."""
    return bool(eval(check, {"re": re, "reply": reply}))


def main(path, model=DEFAULT_MODEL):
    with open(path) as f:
        R = json.load(f)
    system, queries, check = R["system"], R["queries"], R.get("check")
    arms = ("base", "edited", "prompted")
    for a in arms:
        assert len(R[a]) == len(queries), f"{a}: {len(R[a])} replies for {len(queries)} queries"
    flat = [{"query": q, "reply": r} for a in arms for q, r in zip(queries, R[a])]
    t = time.time()
    s = judge(system, flat, model)
    n = len(queries)
    J = {a: s[i * n:(i + 1) * n] for i, a in enumerate(arms)}
    out = {"model": model, "n": n,
           "judge_mean": {a: (None if all(x is None for x in J[a]) else round(_mean(J[a]), 4)) for a in arms},
           "judge_failed": sum(x is None for x in s)}
    g = gap_closed(J["base"], J["edited"], J["prompted"])
    out["judge_gap_closed"] = None if g is None else round(g, 4)
    if check:
        C = {a: [float(run_check(check, r)) for r in R[a]] for a in arms}
        out["check_rate"] = {a: round(_mean(C[a]), 4) for a in arms}
        g = gap_closed(C["base"], C["edited"], C["prompted"])
        out["check_gap_closed"] = None if g is None else round(g, 4)
    print(f"judge model: {model}   n={n} per arm   {time.time() - t:.1f}s   cost ${USAGE['cost']:.4f}")
    print(f"{'':10}{'base':>9}{'edited':>9}{'prompted':>10}{'gap_closed':>12}")
    fmt = lambda d, g: "".join(f"{'n/a' if d[a] is None else format(d[a], '.3f'):>{w}}" for a, w in zip(arms, (9, 9, 10))) + f"{'n/a' if g is None else format(g, '.3f'):>12}"
    print(f"{'judge':10}" + fmt(out["judge_mean"], out["judge_gap_closed"]))
    if check:
        print(f"{'check':10}" + fmt(out["check_rate"], out["check_gap_closed"]))
    if out["judge_failed"]:
        print(f"WARNING: {out['judge_failed']} judgments failed and were excluded from means")
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python3 evalkit/judge.py results.json [model]")
    main(sys.argv[1], *(sys.argv[2:3]))
