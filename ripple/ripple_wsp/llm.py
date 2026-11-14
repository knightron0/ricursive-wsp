import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


OPENROUTER = "https://openrouter.ai/api"
REMOTE_MODEL = "ricursiveai/ripple-65b-preview"
RIPPLE_PROVIDERS = ["primeintellect", "together"]
NO_KEY = "no OpenRouter key: set OPENROUTER_API_KEY, or OPENROUTER_KEY_FILE to a file holding it"


def remote_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    path = os.environ.get("OPENROUTER_KEY_FILE")
    try:
        return Path(path).expanduser().read_text().strip() if path else ""
    except OSError:
        return ""


FATAL = {400, 401, 402, 403, 404}
GIVE_UP_SECONDS = 6 * 3600


class FileAgent:
    def __init__(self, box: Path, label: str = "external agent", poll: float = 10.0, give_up: float = 8 * 3600):
        self.box, self.poll, self.give_up, self.n = Path(box), poll, give_up, 0
        self.box.mkdir(parents=True, exist_ok=True)
        self.settings = {"model": label, "backend": "file", "box": str(self.box)}

    def chat(self, messages: list[dict], live: Path) -> dict:
        self.n += 1
        inbox, outbox = self.box / f"in_{self.n:03d}.md", self.box / f"out_{self.n:03d}.md"
        tmp = inbox.with_suffix(".tmp")
        tmp.write_text(messages[-1]["content"])
        tmp.rename(inbox)
        print(f"wsp: waiting for {outbox.name}", flush=True)
        t0 = time.time()
        while not outbox.exists():
            if time.time() - t0 > self.give_up:
                raise RuntimeError(f"no reply in {outbox} after {self.give_up / 3600:.0f} h")
            time.sleep(self.poll)
        content = outbox.read_text()
        live.write_text(content)
        return {
            "content": content, "reasoning": "", "finish_reason": "stop", "seconds": time.time() - t0, "provider": "file",
            "prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None, "cached_tokens": None, "tokens_per_second": 0.0,
            "cost": None, "retries": [],
        }


class OpenRouter:
    def __init__(self, model: str, providers: list[str], reasoning_budget: int, max_tokens: int, temperature: float, top_p: float):
        key = remote_key()
        if not key:
            raise SystemExit(f"wsp: {NO_KEY}")
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        self.settings = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning": {"max_tokens": reasoning_budget},
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if providers:
            self.settings["provider"] = {"order": providers, "only": providers, "require_parameters": True}

    def chat(self, messages: list[dict], live: Path) -> dict:
        body = json.dumps({**self.settings, "messages": messages}).encode()
        t_start, attempt, retries = time.time(), 0, []
        while True:
            attempt += 1
            try:
                return self._stream(body, live) | {"retries": retries}
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:600]
                if e.code in FATAL:
                    raise RuntimeError(f"OpenRouter refused the request: HTTP {e.code} {detail}")
                why, wait = f"HTTP {e.code}", 20 if e.code == 429 else 30
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
                why, wait = repr(e)[:300], 30
            retries.append(why)
            if time.time() - t_start > GIVE_UP_SECONDS:
                raise RuntimeError(f"OpenRouter unavailable for {GIVE_UP_SECONDS // 3600} h; last error {why}")
            print(f"wsp: chat attempt {attempt} failed: {why}; retrying in {wait}s", flush=True)
            time.sleep(wait)

    def _stream(self, body: bytes, live: Path) -> dict:
        req = urllib.request.Request(f"{OPENROUTER}/v1/chat/completions", data=body, headers=self.headers)
        content, reasoning, finish, usage, provider, t_first = [], [], None, None, None, None
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=900) as r, live.open("w") as lf:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise ConnectionError(f"provider error mid-stream: {chunk['error']}")
                usage = chunk.get("usage") or usage
                provider = chunk.get("provider") or provider
                if not chunk.get("choices"):
                    continue
                choice = chunk["choices"][0]
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                piece = delta.get("reasoning") or delta.get("reasoning_content")
                if (piece or delta.get("content")) and t_first is None:
                    t_first = time.time()
                if piece:
                    reasoning.append(piece)
                    lf.write(piece)
                if delta.get("content"):
                    content.append(delta["content"])
                    lf.write(delta["content"])
                lf.flush()
        if finish is None:
            raise ConnectionError(f"stream ended without a finish_reason after {len(content) + len(reasoning)} chunks")
        usage = usage or {}
        n = usage.get("completion_tokens") or 0
        gen = max(time.time() - (t_first or t0), 1e-6)
        return {
            "content": "".join(content),
            "reasoning": "".join(reasoning),
            "finish_reason": finish,
            "seconds": time.time() - t0,
            "provider": provider,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": n,
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "tokens_per_second": n / gen,
            "cost": usage.get("cost"),
        }
