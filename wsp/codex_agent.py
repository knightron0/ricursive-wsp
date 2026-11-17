"""Run the episode through the Codex Python SDK (`openai-codex`, which drives `codex app-server`).

The episode's calls are served by a small MCP server (streamable HTTP, JSON responses) running in this process on
127.0.0.1 under a random path; Codex is pointed at it with config overrides. Codex's own tools (shell, apply_patch,
web search, image viewing, sub-agents, plugins, ...) are switched off, its base instructions are replaced by our
system prompt, and it runs with a private CODEX_HOME (only auth.json is linked in), so no user config, AGENTS.md,
skills, memories or hooks reach the model. Any item in the transcript other than messages, reasoning and calls to
our server is reported by the audit.
"""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import threading
import time

from wsp.sdk_agent import hold_for_usage, to_json, turn_note

SERVER = 'wsp'
MCP_PROTOCOLS = ('2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05')
TOOL_OUTPUT_TOKENS = 200_000

CODEX_SYSTEM = f"""Each call above is a tool of the MCP server `{SERVER}`: data, labels, tools, end_tools, predict,
evaluate, submit. There are no other tools."""

# Item types a clean transcript may hold; anything else means a Codex tool leaked through.
ALLOWED_ITEMS = {'userMessage', 'agentMessage', 'reasoning', 'mcpToolCall', 'contextCompaction', 'plan'}

# Codex features that would give the model tools beyond the episode's calls.
DISABLED_FEATURES = ('shell_tool', 'unified_exec', 'view_image', 'image_generation', 'multi_agent', 'multi_agent_v2',
                     'apps', 'plugins', 'remote_plugin', 'browser_use', 'browser_use_external', 'in_app_browser',
                     'computer_use', 'sleep_tool', 'tool_suggest', 'skill_search', 'skill_mcp_dependency_install',
                     'goals', 'hooks', 'memories', 'shell_snapshot')
# Keep code_mode_host enabled: the Codex runtime needs it to dispatch our MCP tools.


def toml_value(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(toml_value(v) for v in value) + ']'
    return json.dumps(value)


def config_overrides(url, tool_timeout, max_tokens=None):
    cfg = {f'features.{f}': False for f in DISABLED_FEATURES}
    cfg.update({
        'web_search': 'disabled', 'tools.view_image': False, 'include_apply_patch_tool': False,
        'project_doc_max_bytes': 0, 'notify': [], 'hide_agent_reasoning': False, 'check_for_update_on_startup': False,
        'tool_output_token_limit': TOOL_OUTPUT_TOKENS,
        f'mcp_servers.{SERVER}.url': url, f'mcp_servers.{SERVER}.tool_timeout_sec': tool_timeout,
        f'mcp_servers.{SERVER}.startup_timeout_sec': 60,
        f'mcp_servers.{SERVER}.default_tools_approval_mode': 'approve',
    })
    if max_tokens:
        cfg['model_max_output_tokens'] = max_tokens
    return tuple(f'{k}={toml_value(v)}' for k, v in cfg.items())


def private_home(path, source=None):
    """A CODEX_HOME holding only a link to the user's auth.json (so refreshed tokens stay shared)."""
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    source = Path(source or os.environ.get('CODEX_HOME') or Path.home() / '.codex')
    auth = path / 'auth.json'
    if (source / 'auth.json').exists() and not auth.exists():
        auth.symlink_to((source / 'auth.json').resolve())
    return path


def five_hour_window(snapshot):
    """(fraction used, reset time) of the 300-minute window in a rate-limit snapshot (dict or model)."""
    snapshot = to_plain(snapshot)
    for key in ('primary', 'secondary'):
        w = (snapshot or {}).get(key) or {}
        if w.get('windowDurationMins') == 300 and w.get('usedPercent') is not None:
            return w['usedPercent'] / 100, w.get('resetsAt')
    return None, None


def to_plain(obj):
    if hasattr(obj, 'model_dump'):
        return obj.model_dump(mode='json', by_alias=True, exclude_none=True)
    return to_json(obj)


def usage_entry(reply, last):
    """One reply's token usage, in the Responses API's field names (what the run log already prints)."""
    last = to_plain(last)
    return {'reply': reply, 'input_tokens': last.get('inputTokens', 0),
            'input_tokens_details': {'cached_tokens': last.get('cachedInputTokens', 0)},
            'output_tokens': last.get('outputTokens', 0),
            'output_tokens_details': {'reasoning_tokens': last.get('reasoningOutputTokens', 0)}}


def content_types(messages):
    """Block types for the audit: item types, MCP result content types, 'forbidden:<what>' for leaked tools."""
    types = []
    for m in messages:
        if m.get('event') != 'item/completed':
            continue
        item = m.get('item') or {}
        kind = item.get('type')
        if kind not in ALLOWED_ITEMS:
            types.append(f'forbidden:{kind}')
            if kind == 'imageView':
                types.append('image')
            continue
        types.append(kind)
        if kind == 'mcpToolCall':
            if item.get('server') != SERVER:
                types.append(f'forbidden:mcp server {item.get("server")}')
            types += [c.get('type') for c in ((item.get('result') or {}).get('content') or []) if isinstance(c, dict)]
    return types


class EpisodeMCP:
    """The episode's calls as an MCP server over streamable HTTP; one call runs at a time."""

    def __init__(self, episode, specs, turns=None, window=None, on_close=None):
        self.episode, self.specs, self.turns, self.window, self.on_close = episode, specs, turns, window, on_close
        self.lock = threading.Lock()
        self.path = f'/mcp/{secrets.token_urlsafe(16)}'
        self.httpd = None

    def tool_list(self):
        return [{'name': s['name'], 'description': s['description'], 'inputSchema': s['input_schema']}
                for s in self.specs]

    def call(self, name, args):
        """Returns (text, is_error), with the turn note appended while a turn limit is set."""
        turns = self.turns
        turn = turns['reply'] + 1 if turns else None
        with self.lock:
            asyncio.run(hold_for_usage(self.window))
            if self.episode.submission is not None:
                return 'The episode is over: submit() was accepted.', True
            if turns and turns['limit'] and turn > turns['limit']:
                turns['closed'] = True
            if turns and turns['closed']:
                if self.on_close:
                    self.on_close()
                return 'The episode is over: the turn limit was reached.', True
            if name not in {s['name'] for s in self.specs}:
                return f'{name} is not available in this episode', True
            text, error = self.episode.call(name, dict(args or {}))
        if turns and turns['limit'] and self.episode.submission is None:
            text += turn_note({'reply': turn, 'limit': turns['limit']})
        return text, error

    def handle(self, msg):
        """One JSON-RPC message -> response dict, or None for notifications and responses."""
        if not isinstance(msg, dict) or 'method' not in msg:
            return None
        method, params, mid = msg['method'], msg.get('params') or {}, msg.get('id')
        if mid is None:
            return None
        if method == 'initialize':
            asked = params.get('protocolVersion')
            result = {'protocolVersion': asked if asked in MCP_PROTOCOLS else MCP_PROTOCOLS[0],
                      'capabilities': {'tools': {'listChanged': False}},
                      'serverInfo': {'name': SERVER, 'version': '1'}}
        elif method == 'ping':
            result = {}
        elif method == 'tools/list':
            result = {'tools': self.tool_list()}
        elif method == 'tools/call':
            text, error = self.call(params.get('name'), params.get('arguments'))
            result = {'content': [{'type': 'text', 'text': text}], 'isError': bool(error)}
        else:
            return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32601, 'message': f'unknown method {method}'}}
        return {'jsonrpc': '2.0', 'id': mid, 'result': result}

    def start(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def reply(self, code, body=None):
                data = json.dumps(body).encode() if body is not None else b''
                self.send_response(code)
                if body is not None:
                    self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                if self.path != server.path:
                    return self.reply(404)
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'null')
                except ValueError:
                    return self.reply(400, {'jsonrpc': '2.0', 'id': None,
                                            'error': {'code': -32700, 'message': 'parse error'}})
                if isinstance(body, list):
                    out = [r for r in map(server.handle, body) if r is not None]
                else:
                    out = server.handle(body)
                return self.reply(200, out) if out else self.reply(202)

            def do_GET(self):
                self.reply(405)

            def do_DELETE(self):
                self.reply(200 if self.path == server.path else 404)

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return f'http://127.0.0.1:{self.httpd.server_address[1]}{self.path}'

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()


class CodexRun:
    def __init__(self, episode, system, prompt, specs, model, cwd, effort=None, max_turns=50, max_tokens=None,
                 on_reply=None, max_replies=None, usage_limit=None, tool_timeout=300, codex_bin=None):
        self.episode, self.system, self.prompt, self.specs = episode, system, prompt, specs
        self.model, self.cwd, self.effort, self.max_turns, self.max_tokens = model, Path(cwd).resolve(), effort, max_turns, max_tokens
        self.on_reply, self.tool_timeout, self.codex_bin = on_reply, tool_timeout, codex_bin
        self.turns = {'reply': 0, 'limit': max_replies, 'closed': False}
        self.window = {'limit': usage_limit, 'used': None, 'resets': None, 'holds': []}
        self.messages, self.usage, self.result = [{'event': 'prompt', 'system': system, 'user': prompt}], [], None
        self.stop = None

    def mcp_timeout(self):
        # A call can wait out a usage hold; otherwise allow well beyond the tools() timeout.
        return 6 * 3600 if self.window['limit'] else max(30 * 60, 2 * self.tool_timeout)

    def rate_event(self, snapshot):
        used, resets = five_hour_window(snapshot)
        if used is not None:
            self.window['used'], self.window['resets'] = used, resets

    def token_event(self, payload, started):
        """A thread/tokenUsage/updated notification marks the end of one model response."""
        usage = to_plain(payload).get('tokenUsage') or {}
        total = (usage.get('total') or {}).get('totalTokens')
        if self.usage and total is not None and total == self.usage[-1].get('thread_total_tokens'):
            return False
        self.turns['reply'] += 1
        entry = usage_entry(self.turns['reply'], usage.get('last') or {})
        self.usage.append({**entry, 'thread_total_tokens': total})
        if self.on_reply:
            self.on_reply(time.monotonic() - started, entry)
        limit = self.turns['limit']
        if limit and self.turns['reply'] > limit and self.episode.submission is None:
            self.turns['closed'] = True
        return self.turns['closed'] or self.turns['reply'] >= self.max_turns

    def record(self, event):
        if 'delta' in event.method.lower():
            return
        self.messages.append({'event': event.method, **(to_plain(event.payload) or {})})

    def request_stop(self):
        loop, stop = getattr(self, 'loop', None), self.stop
        if loop and stop:
            loop.call_soon_threadsafe(stop.set)

    async def watch_rate_limits(self, client):
        from openai_codex.generated.v2_all import AccountRateLimitsUpdatedNotification
        while True:
            try:
                event = await client.next_notification()
            except Exception:
                return
            if isinstance(event.payload, AccountRateLimitsUpdatedNotification):
                self.rate_event(event.payload.rate_limits)

    async def run(self):
        from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox
        from openai_codex.generated.v2_all import (GetAccountRateLimitsResponse, ReasoningEffort, ReasoningSummary,
                                                   ThreadTokenUsageUpdatedNotification, TurnCompletedNotification)
        self.cwd.mkdir(parents=True, exist_ok=True)
        self.loop, self.stop = asyncio.get_running_loop(), asyncio.Event()
        mcp = EpisodeMCP(self.episode, self.specs, self.turns, self.window, on_close=self.request_stop)
        url = mcp.start()
        home = private_home(self.cwd.parent / 'codex_home')
        config = CodexConfig(codex_bin=self.codex_bin, cwd=str(self.cwd), env={'CODEX_HOME': str(home)},
                             config_overrides=config_overrides(url, self.mcp_timeout(), self.max_tokens))
        watcher = None
        try:
            async with AsyncCodex(config) as codex:
                if self.window['limit']:
                    try:
                        limits = await codex._client.request('account/rateLimits/read', None,
                                                             response_model=GetAccountRateLimitsResponse)
                        self.rate_event(limits.rate_limits)
                    except Exception as e:
                        print(f'could not read rate limits: {e}', flush=True)
                    watcher = asyncio.create_task(self.watch_rate_limits(codex._client))
                thread = await codex.thread_start(
                    approval_mode=ApprovalMode.deny_all, sandbox=Sandbox.read_only, cwd=str(self.cwd),
                    base_instructions=f'{self.system}\n\n{CODEX_SYSTEM}', model=self.model, ephemeral=False)
                turn = await thread.turn(self.prompt, effort=ReasoningEffort(self.effort) if self.effort else None,
                                         summary=ReasoningSummary('detailed'))
                interrupted = False

                async def interrupt():
                    nonlocal interrupted
                    if not interrupted:
                        interrupted = True
                        await turn.interrupt()

                async def on_stop():
                    await self.stop.wait()
                    await interrupt()

                stopper = asyncio.create_task(on_stop())
                started = time.monotonic()
                try:
                    async for event in turn.stream():
                        self.record(event)
                        payload = event.payload
                        if isinstance(payload, ThreadTokenUsageUpdatedNotification):
                            if self.token_event(payload, started):
                                await interrupt()
                            started = time.monotonic()
                        elif isinstance(payload, TurnCompletedNotification):
                            done = to_plain(payload.turn)
                            self.result = {'status': done.get('status'), 'error': done.get('error'),
                                           'duration_ms': done.get('durationMs'), 'replies': self.turns['reply'],
                                           'interrupted': interrupted}
                        if self.episode.submission is not None and event.method == 'item/completed':
                            await interrupt()
                finally:
                    stopper.cancel()
        finally:
            mcp.stop()
            if watcher:
                watcher.cancel()

    def transcript(self):
        return self.messages

    def content_types(self):
        return content_types(self.messages)
