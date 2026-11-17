import asyncio
import dataclasses
import os
from pathlib import Path
import time

SERVER = 'wsp'
MAX_RESULT_CHARS = 500_000

SDK_SYSTEM = """Each call above is a tool named mcp__wsp__<call>: mcp__wsp__data, mcp__wsp__labels, mcp__wsp__tools,
mcp__wsp__end_tools, mcp__wsp__predict, mcp__wsp__evaluate, mcp__wsp__submit. If a tool result is too long to show
inline and is saved to a file instead, open it with Read; Read works only on those saved tool outputs."""


def tool_name(call):
    return f'mcp__{SERVER}__{call}'


def saved_outputs_dir(cwd):
    from claude_agent_sdk import project_key_for_directory
    config = Path(os.environ.get('CLAUDE_CONFIG_DIR', Path.home() / '.claude'))
    return (config / 'projects' / project_key_for_directory(cwd)).resolve()


def read_allowed(path, cwd):
    real = Path(os.path.realpath(path))
    root = saved_outputs_dir(cwd)
    return real.is_relative_to(root) and 'tool-results' in real.relative_to(root).parts


def to_json(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {'type': type(obj).__name__, **{f.name: to_json(getattr(obj, f.name)) for f in dataclasses.fields(obj)}}
    if isinstance(obj, (list, tuple)):
        return [to_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    return obj


def content_types(messages):
    names = {'TextBlock': 'text', 'ThinkingBlock': 'thinking', 'ToolUseBlock': 'tool_use',
             'ToolResultBlock': 'tool_result'}
    types = []
    for m in messages:
        content = m.get('content')
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            types.append(names.get(block.get('type'), block.get('type')))
            if isinstance(block.get('content'), list):
                types += [c.get('type') for c in block['content'] if isinstance(c, dict)]
    return types


def build_server(episode, specs, turns=None, window=None):
    from claude_agent_sdk import create_sdk_mcp_server
    return create_sdk_mcp_server(SERVER, tools=episode_tools(episode, specs, turns, window))


def five_hour_window(event):
    raw = event.rate_limit_info.raw if hasattr(event, 'rate_limit_info') else event['rate_limit_info']['raw']
    w = (raw or {}).get('unifiedWindows', {}).get('five_hour', {})
    return w.get('utilization'), w.get('resetsAt')


async def hold_for_usage(window, sleep=None, now=None):
    sleep, now = sleep or asyncio.sleep, now or time.time
    while window and window['limit'] and window['used'] is not None and window['used'] >= window['limit']:
        resets = window['resets'] or now() + 600
        seconds = max(60.0, resets - now() + 120)
        window['holds'].append({'at': now(), 'used': window['used'], 'seconds': seconds})
        print(f'usage hold: five-hour window {window["used"]:.0%} used >= limit {window["limit"]:.0%}; holding the '
              f'next call {seconds:.0f}s, until after the reset', flush=True)
        await sleep(seconds)
        window['used'] = None


def turn_note(turns):
    turn, limit = turns['reply'], turns['limit']
    if turn >= limit:
        return (f'\n[harness] turn {turn} of {limit}: this is your last turn. Without an accepted submit() in this '
                'reply the episode ends with no score.')
    return f'\n[harness] turn {turn} of {limit}; {limit - turn} turns left after this one.'


def episode_tools(episode, specs, turns=None, window=None):
    from claude_agent_sdk import ToolAnnotations, tool
    lock = asyncio.Lock()

    def make(spec):
        name = spec['name']

        async def handler(args):
            async with lock:
                await hold_for_usage(window)
                if episode.submission is not None:
                    return {'content': [{'type': 'text', 'text': 'The episode is over: submit() was accepted.'}],
                            'is_error': True}
                if turns and turns['closed']:
                    return {'content': [{'type': 'text', 'text': 'The episode is over: the turn limit was reached.'}],
                            'is_error': True}
                text, error = await asyncio.to_thread(episode.call, name, dict(args))
            if turns and turns['limit'] and episode.submission is None:
                text += turn_note(turns)
            return {'content': [{'type': 'text', 'text': text}], 'is_error': error}

        return tool(name, spec['description'], spec['input_schema'],
                    annotations=ToolAnnotations(maxResultSizeChars=MAX_RESULT_CHARS))(handler)

    return [make(s) for s in specs]


def read_fence(cwd):
    async def hook(input_data, tool_use_id, context):
        path = (input_data.get('tool_input') or {}).get('file_path', '')
        if read_allowed(path, cwd):
            return {}
        return {'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'permissionDecision': 'deny',
                                       'permissionDecisionReason': 'Read works only on saved tool outputs'}}
    return hook


class SDKRun:
    def __init__(self, episode, system, prompt, specs, model, cwd, effort=None, max_turns=50, max_tokens=None,
                 on_reply=None, max_replies=None, usage_limit=None):
        self.episode, self.system, self.prompt, self.specs = episode, system, prompt, specs
        self.model, self.cwd, self.effort, self.max_turns, self.max_tokens = model, Path(cwd), effort, max_turns, max_tokens
        self.on_reply = on_reply
        self.turns = {'reply': 0, 'limit': max_replies, 'closed': False}
        self.window = {'limit': usage_limit, 'used': None, 'resets': None, 'holds': []}
        self.messages, self.usage, self.result = [], [], None

    def rate_event(self, event):
        used, resets = five_hour_window(event)
        if used is not None:
            self.window['used'], self.window['resets'] = used, resets

    def new_reply(self):
        self.turns['reply'] += 1
        limit = self.turns['limit']
        if limit and self.turns['reply'] > limit and self.episode.submission is None:
            self.turns['closed'] = True
        return self.turns['closed']

    def options(self):
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, PermissionResultDeny

        async def deny_others(name, args, context):
            return PermissionResultDeny(message=f'{name} is not available in this episode')

        env = {'MCP_TOOL_TIMEOUT': str((6 * 60 if self.window['limit'] else 30) * 60 * 1000),
               'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1'}
        if self.max_tokens:
            env['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] = str(self.max_tokens)
        return ClaudeAgentOptions(
            tools=['Read'], allowed_tools=[tool_name(s['name']) for s in self.specs] + ['Read'],
            mcp_servers={SERVER: build_server(self.episode, self.specs, self.turns, self.window)},
            strict_mcp_config=True,
            system_prompt=f'{self.system}\n\n{SDK_SYSTEM}', setting_sources=[], skills=[], model=self.model,
            effort=self.effort, thinking={'type': 'adaptive', 'display': 'summarized'}, max_turns=self.max_turns,
            cwd=str(self.cwd), env=env, can_use_tool=deny_others,
            hooks={'PreToolUse': [HookMatcher(matcher='Read', hooks=[read_fence(self.cwd)])]})

    async def run(self):
        from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, RateLimitEvent, ResultMessage
        self.cwd.mkdir(parents=True, exist_ok=True)
        reply, started, interrupted = None, time.monotonic(), False
        async with ClaudeSDKClient(options=self.options()) as client:
            await client.query(self.prompt)
            async for message in client.receive_response():
                self.messages.append(to_json(message))
                if isinstance(message, AssistantMessage):
                    if message.message_id != reply:
                        if reply is not None:
                            self.reply_done(started)
                            started = time.monotonic()
                        if self.new_reply() and not interrupted:
                            interrupted = True
                            await client.interrupt()
                    reply = message.message_id
                    if message.usage:
                        if self.usage and self.usage[-1].get('message_id') == reply:
                            self.usage[-1] = {'message_id': reply, **message.usage}
                        else:
                            self.usage.append({'message_id': reply, **message.usage})
                    if self.episode.submission is not None and not interrupted:
                        interrupted = True
                        await client.interrupt()
                elif isinstance(message, RateLimitEvent):
                    self.rate_event(message)
                elif isinstance(message, ResultMessage):
                    self.result = to_json(message)
        self.reply_done(started)

    def reply_done(self, started):
        if self.on_reply:
            self.on_reply(time.monotonic() - started, self.usage[-1] if self.usage else None)

    def transcript(self):
        return self.messages

    def content_types(self):
        return content_types(self.messages)
