"""ripple_api.py -- OpenRouter chat call for Ripple with retries, a cost ledger and a hard spend cap.

Key: environment variable OPENROUTER_API_KEY (never written to disk by this project).
Every response's usage.cost is appended to logs/api_cost.jsonl; a call is refused once the
ledger total reaches RIPPLE_SPEND_CAP_USD (default 15).
"""
import json, os, subprocess, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "ricursiveai/ripple-65b-preview"
PROVIDER = {"order": ["GMICloud"], "allow_fallbacks": False}   # structured tool calls work here
LEDGER = os.path.join(HERE, "logs", "api_cost.jsonl")
CAP = float(os.environ.get("RIPPLE_SPEND_CAP_USD", "15"))


class SpendCapReached(RuntimeError):
    pass


def api_key():
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not k:  # local convenience on the build machine; the key is never stored in this folder
        helper = os.path.expanduser("~/.lor/extensions/secure-paste/secret")
        if os.path.exists(helper):
            k = subprocess.run([helper, "OPENROUTER_API_KEY"], capture_output=True, text=True).stdout.strip()
    if not k:
        raise RuntimeError("set OPENROUTER_API_KEY")
    return k


def spent():
    if not os.path.exists(LEDGER):
        return 0.0
    return sum(json.loads(l).get("cost", 0.0) for l in open(LEDGER) if l.strip())


def chat(messages, tools=None, max_tokens=16000, run="", retries=40, temperature=None):
    """Returns the parsed response dict with _latency_s and _cost added."""
    total = spent()
    if total >= CAP:
        raise SpendCapReached(f"spend cap ${CAP} reached (ledger ${total:.4f})")
    payload = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
               "usage": {"include": True}, "provider": PROVIDER}
    if tools:
        payload["tools"] = tools
    if temperature is not None:
        payload["temperature"] = temperature
    key, last = api_key(), ""
    for attempt in range(retries + 1):
        if attempt:
            # GMICloud answers many requests with an immediate 502 in bursts; short, persistent retries get through
            time.sleep(min(20, 2 + 2 * attempt))
        t0 = time.time()
        try:
            req = urllib.request.Request(API_URL, data=json.dumps(payload).encode(), method="POST", headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "ripple-repair"})
            with urllib.request.urlopen(req, timeout=900) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
            if data.get("error") or not data.get("choices"):
                last = "bad body: " + json.dumps(data)[:400]
            elif data["choices"][0].get("error"):
                last = "choice error: " + json.dumps(data["choices"][0]["error"])[:400]
            else:
                u = data.get("usage") or {}
                data["_latency_s"] = round(time.time() - t0, 2)
                data["_cost"] = u.get("cost") or 0.0
                with open(LEDGER, "a") as f:
                    f.write(json.dumps({"t": round(time.time(), 1), "run": run, "cost": data["_cost"],
                                        "prompt_tokens": u.get("prompt_tokens"),
                                        "completion_tokens": u.get("completion_tokens"),
                                        "provider": data.get("provider"), "latency_s": data["_latency_s"]}) + "\n")
                return data
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}"
            if e.code in (400, 401, 403):
                break
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        if attempt < 3 or attempt % 10 == 0:
            print(f"    api error ({last[:120]}); attempt {attempt + 1}/{retries + 1}", flush=True)
    raise RuntimeError(f"API failed: {last}")
