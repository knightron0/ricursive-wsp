"""Agents and the episode loop. An agent's next(results) returns its next list of calls."""
import json


class AnthropicAgent:
    """Claude via the Messages API: streaming, adaptive thinking, client tools, text-only content."""

    def __init__(self, system, prompt, tools, model='claude-sonnet-5-5', max_tokens=64000, effort=None):
        import anthropic
        self.client = anthropic.Anthropic(timeout=3600, max_retries=8)
        self.system, self.tools, self.model, self.max_tokens = system, tools, model, max_tokens
        self.output_config = {'effort': effort} if effort else {}
        self.messages = [{'role': 'user', 'content': prompt}]
        self.usage = []

    def next(self, results):
        if results is not None:
            self.messages.append({'role': 'user', 'content': results if isinstance(results, str) else [
                {'type': 'tool_result', 'tool_use_id': i, 'content': text, 'is_error': err} for i, text, err in results]})
        # On a policy refusal the API re-runs the request on a fallback model ("default") in the same call.
        with self.client.beta.messages.stream(
            model=self.model, max_tokens=self.max_tokens, system=self.system, tools=self.tools,
            messages=self.messages, thinking={'type': 'adaptive'}, cache_control={'type': 'ephemeral'},
            betas=['server-side-fallback-2026-07-01'], fallbacks='default',
            **({'output_config': self.output_config} if self.output_config else {}),
        ) as stream:
            message = stream.get_final_message()
        self.messages.append({'role': 'assistant', 'content': message.content})
        self.usage.append(message.usage.model_dump())
        if message.stop_reason in ('refusal', 'max_tokens'):
            raise RuntimeError(f'reply stopped with {message.stop_reason}')
        calls = []
        for b in message.content:
            if b.type == 'tool_use':
                args = b.input
                if not isinstance(args, dict):  # eager input streaming: the client validates
                    args = {}
                calls.append((b.id, b.name, args))
        return calls

    def transcript(self):
        return [m if isinstance(m['content'], str) else
                {'role': m['role'], 'content': [b if isinstance(b, dict) else b.model_dump() for b in m['content']]}
                for m in self.messages]

    def content_types(self):
        """Every content-block type in the conversation (the audit checks no image ever appears)."""
        types = []
        for m in self.messages:
            if isinstance(m['content'], str):
                types.append('text')
                continue
            for b in m['content']:
                types.append(b['type'] if isinstance(b, dict) else b.type)
        return types


class OpenAIAgent:
    """OpenAI via the Responses API: function tools, text-only input, stateless (store=False)."""

    def __init__(self, system, prompt, tools, model='gpt-6.1-sol', max_tokens=64000, effort=None):
        from openai import OpenAI
        self.client = OpenAI(timeout=3600, max_retries=8)
        self.system, self.model, self.max_tokens = system, model, max_tokens
        self.reasoning = {'effort': effort} if effort else None
        self.tools = [{'type': 'function', 'name': t['name'], 'description': t['description'],
                       'parameters': t['input_schema'], 'strict': True} for t in tools]
        self.input = [{'role': 'user', 'content': prompt}]
        self.usage = []

    def next(self, results):
        if isinstance(results, str):
            self.input.append({'role': 'user', 'content': results})
        elif results is not None:
            self.input += [{'type': 'function_call_output', 'call_id': i, 'output': f'ERROR: {text}' if err else text}
                           for i, text, err in results]
        response = self.client.responses.create(
            model=self.model, instructions=self.system, input=self.input, tools=self.tools, store=False,
            include=['reasoning.encrypted_content'], max_output_tokens=self.max_tokens,
            **({'reasoning': self.reasoning} if self.reasoning else {}))
        if response.status != 'completed':
            raise RuntimeError(f'response {response.status}: {response.incomplete_details}')
        self.input += [item.model_dump(exclude_none=True) for item in response.output]
        self.usage.append(response.usage.model_dump() if response.usage else None)
        calls = []
        for item in response.output:
            if item.type == 'function_call':
                try:
                    args = json.loads(item.arguments)
                except json.JSONDecodeError:
                    args = {}
                calls.append((item.call_id, item.name, args if isinstance(args, dict) else {}))
        return calls

    def transcript(self):
        return self.input

    def content_types(self):
        types = []
        for item in self.input:
            content = item.get('content')
            if isinstance(content, list):
                types += [c.get('type') for c in content if isinstance(c, dict)]
            else:
                types.append(item.get('type', 'message'))
        return types


class OpenRouterAgent:
    """Any OpenRouter model (default Claude Opus 5.5) via its chat-completions API: function tools, text only.
    Reasoning is passed back each turn (reasoning_details) so thinking carries across tool calls, and the
    system prompt plus the newest message carry cache_control breakpoints (Anthropic prompt caching)."""

    def __init__(self, system, prompt, tools, model='anthropic/claude-opus-5.5', max_tokens=64000, effort=None):
        import os
        from openai import OpenAI
        self.client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=os.environ['OPENROUTER_API_KEY'],
                             timeout=3600, max_retries=8)
        self.model, self.max_tokens = model, max_tokens
        self.reasoning = {'effort': effort} if effort else {'enabled': True}
        self.tools = [{'type': 'function', 'function': {'name': t['name'], 'description': t['description'],
                                                        'parameters': t['input_schema']}} for t in tools]
        self.system = {'role': 'system', 'content': [{'type': 'text', 'text': system,
                                                      'cache_control': {'type': 'ephemeral'}}]}
        self.messages = [{'role': 'user', 'content': prompt}]
        self.usage = []

    def _request_messages(self):
        last = dict(self.messages[-1])
        if isinstance(last.get('content'), str):
            last['content'] = [{'type': 'text', 'text': last['content'], 'cache_control': {'type': 'ephemeral'}}]
        return [self.system, *self.messages[:-1], last]

    def next(self, results):
        if isinstance(results, str):
            self.messages.append({'role': 'user', 'content': results})
        elif results is not None:
            self.messages += [{'role': 'tool', 'tool_call_id': i, 'content': f'ERROR: {text}' if err else text}
                              for i, text, err in results]
        response = self.client.chat.completions.create(
            model=self.model, messages=self._request_messages(), tools=self.tools, max_tokens=self.max_tokens,
            extra_body={'reasoning': self.reasoning, 'usage': {'include': True}})
        if not response.choices:
            raise RuntimeError(f'no choices: {response}')
        choice = response.choices[0]
        message = choice.message.model_dump(exclude_none=True)
        message = {k: message[k] for k in ('role', 'content', 'tool_calls', 'reasoning', 'reasoning_details') if k in message}
        self.messages.append(message)
        self.usage.append(response.usage.model_dump() if response.usage else None)
        if choice.finish_reason in ('length', 'content_filter'):
            raise RuntimeError(f'reply stopped with {choice.finish_reason}')
        calls = []
        for c in message.get('tool_calls') or []:
            try:
                args = json.loads(c['function']['arguments'] or '{}')
            except json.JSONDecodeError:
                args = {}
            calls.append((c['id'], c['function']['name'], args if isinstance(args, dict) else {}))
        return calls

    def transcript(self):
        return [self.system, *self.messages]

    def content_types(self):
        types = []
        for m in self.messages:
            content = m.get('content')
            if isinstance(content, list):
                types += [c.get('type') for c in content if isinstance(c, dict)]
            else:
                types.append('text')
        return types


class ScriptedAgent:
    """Offline agent: replays a JSON list of turns, each a list of {"name", "args"}."""

    def __init__(self, turns):
        self.turns, self.usage = list(turns), []

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def next(self, results):
        if not self.turns:
            return []
        return [(f'call_{len(self.turns)}_{k}', c['name'], c.get('args', {})) for k, c in enumerate(self.turns.pop(0))]

    def transcript(self):
        return []

    def content_types(self):
        return ['text']


def run_episode(agent, episode, max_turns=40, on_turn=None):
    results = None
    for turn in range(1, max_turns + 1):
        calls = agent.next(results)
        if not calls:
            results = ('Call end_tools(), then submit() with your diff to finish.'
                       if episode.mode == 'reasoned' and not episode.locked else 'Call submit() with your diff to finish.')
            if isinstance(agent, ScriptedAgent) and not agent.turns:
                return
            continue
        results = []
        for call_id, name, args in calls:
            text, error = episode.call(name, args)
            results.append((call_id, text, error))
            if episode.submission is not None:
                break
        if on_turn:
            on_turn(turn)
        if episode.submission is not None:
            return
