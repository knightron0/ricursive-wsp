"""Weight-space programming: an agent writes a literal diff over an initialized ResNet-18.

    python -m wsp --task scratch-cifar10 --mode blind
    python -m wsp --task scratch-cifar10 --mode reasoned --data-fraction 0.1
    python -m wsp --task scratch-cifar10 --mode full     --data-fraction 0.1 --evaluate-test
    python -m wsp ... --dry-run                      # write prompt + tool definitions, no API call
    python -m wsp ... --script calls.json            # replay scripted calls, no API call

Modes (see wsp/episode.py): blind = tools() only, no data; reasoned = phase 1 tools/data/predict
without labels, then end_tools() permanently disables tools() and opens labels/evaluate/submit; full = everything. Val (and test, with
--evaluate-test) are scored once on the submitted diff and never shown to the agent.
"""
import argparse
import asyncio
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
import sys
import time

import torch

from wsp import data as datasets
from wsp.agent import AnthropicAgent, OpenAIAgent, OpenRouterAgent, ScriptedAgent, run_episode
from wsp.episode import MODES, Episode, audit
from wsp.evaluation import evaluate, summarize
from wsp.model import default_device
from wsp.prompt import system_prompt, tool_specs, user_prompt
from wsp.tasks import TASKS, apply_diff, load_start, start_weights, writable_names


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')


def describe(entry):
    first = entry['result'].split('\n', 1)[0]
    if entry.get('status', 'ok') != 'ok':
        return f'{entry["call"]} ({entry["status"]}: {(entry.get("reason") or first)[:160]})'
    if entry['call'] == 'tools' and first.startswith(('exit code', 'timed out')):
        return f'tools ({first})'
    return entry['call']


def usage_summary(usage):
    if not usage:
        return None
    if 'cache_read_input_tokens' in usage or 'cache_creation_input_tokens' in usage:
        cached = usage.get('cache_read_input_tokens') or 0
        total = (usage.get('input_tokens') or 0) + cached + (usage.get('cache_creation_input_tokens') or 0)
    else:
        details = usage.get('prompt_tokens_details') or usage.get('input_tokens_details') or {}
        cached = details.get('cached_tokens') or 0
        total = usage.get('prompt_tokens', usage.get('input_tokens')) or 0
    out = usage.get('output_tokens', usage.get('completion_tokens')) or 0
    parts = [f'in {total / 1000:.1f}k ({cached / 1000:.1f}k cached)', f'out {out / 1000:.1f}k']
    if usage.get('cost') is not None:
        parts.append(f'${usage["cost"]:.3f}')
    return ', '.join(parts)


def pending_block(results):
    if isinstance(results, str):
        return {'role': 'pending_user_message', 'content': results}
    return {'role': 'pending_tool_results',
            'content': [{'tool_call_id': i, 'content': text, 'is_error': error} for i, text, error in results]}


def turn_line(turn, entries, mode, locked, seconds, usage, counted=True):
    head = f'turn {turn}' if counted else f'turn {turn + 1} retry (format only, not counted)'
    if mode == 'reasoned':
        end = 2 if locked else 1
        start = entries[0]['phase'] if entries else end
        head += f' [phase {start}]' if start == end else f' [phase {start}->{end}]'
    tail = [f'{seconds:.0f}s'] + [s for s in [usage_summary(usage)] if s]
    return f'{head}: {", ".join(map(describe, entries)) or "no calls"} | {" | ".join(tail)}'


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog='python -m wsp', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', choices=sorted(TASKS), required=True)
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--data-fraction', type=float, default=0.0, help='Fraction (0, 1] of train (reasoned/full)')
    p.add_argument('--data-seed', type=int, default=0)
    p.add_argument('--provider', choices=['anthropic', 'openai', 'openrouter', 'claude-sdk'], default='anthropic')
    p.add_argument('--model', help='default: claude-sonnet-5-5 (anthropic), gpt-6.1-sol (openai), '
                                   'anthropic/claude-opus-5.5 (openrouter), claude-opus-5-5 (claude-sdk)')
    p.add_argument('--token-budget', type=int, default=960000,
                   help='claude-sdk: soft total output-token budget stated in the prompt (not enforced)')
    p.add_argument('--sdk-max-turns', type=int,
                   help="claude-sdk: backstop on Claude Code's own agent-loop turns (default 3 x --max-turns, else 50)")
    p.add_argument('--effort', choices=['low', 'medium', 'high', 'xhigh', 'max'])
    p.add_argument('--max-tokens', type=int, default=64000, help='Per model turn')
    p.add_argument('--max-turns', type=int,
                   help='Model replies (default 10; format-only rejections do not count). With claude-sdk: no limit '
                        'unless given; then every reply counts, the limit is enforced and stated in the prompt')
    p.add_argument('--format-retries', type=int, default=3,
                   help='Most replies that may go uncounted because every call in them was a format rejection')
    p.add_argument('--max-numbers', type=int, default=20000, help='Most numbers a diff may hold')
    p.add_argument('--tool-timeout', type=float, default=300)
    p.add_argument('--script', type=Path, help='Replay scripted calls instead of calling the API')
    p.add_argument('--source', type=Path, default=Path('checkpoints/resnet18_cifar10.pt'))
    p.add_argument('--start-weights', type=Path,
                   help='Start from this state_dict (another agent\'s submitted network); the prompt says so')
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    p.add_argument('--device', default=default_device())
    p.add_argument('--output', type=Path,
                   default=Path('runs') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--evaluate-test', action='store_true')
    a = p.parse_args(argv)
    if (a.mode == 'blind') != (a.data_fraction == 0) or not 0 <= a.data_fraction <= 1:
        p.error('blind takes --data-fraction 0; reasoned/full need --data-fraction in (0, 1]')
    if a.max_turns is None and (a.provider != 'claude-sdk' or a.script):
        a.max_turns = 10
    if a.sdk_max_turns is None:
        a.sdk_max_turns = 3 * a.max_turns if a.max_turns else 50
    return a


def load_dotenv(path=Path(__file__).resolve().parent.parent / '.env'):
    """KEY=value lines from the repo's .env (gitignored); variables already set win."""
    if path.exists():
        for line in path.read_text().splitlines():
            key, eq, value = line.partition('=')
            key = key.strip().removeprefix('export ').strip()
            if eq and key.isidentifier() and not line.lstrip().startswith('#'):
                os.environ.setdefault(key, value.strip().strip('"\''))


def main(argv=None):
    load_dotenv()
    a = parse_args(argv)
    task = TASKS[a.task]
    if task.needs_source and not a.source.exists():
        sys.exit(f'{a.source} not found; run `python train_base.py` first')
    if a.start_weights:
        start = load_start(a.start_weights, task.num_classes)
    else:
        start = start_weights(task, torch.load(a.source, weights_only=True) if task.needs_source else None)
    classes = datasets.class_names(task.dataset, a.data_dir)
    images, labels = datasets.load_split(task.dataset, 'train', a.data_dir)
    subset = (torch.zeros(0, dtype=torch.long) if a.mode == 'blind'
              else datasets.stratified_subset(labels, a.data_fraction, a.data_seed))
    images, labels = images[subset].clone(), labels[subset].clone()
    start_score = (evaluate(start, datasets.load_split(task.dataset, 'val', a.data_dir), classes, a.device)['accuracy']
                   if a.start_weights else None)

    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'config.json', vars(a))
    save(out / 'subset_indices.json', {'indices': subset.tolist()})
    sdk = a.provider == 'claude-sdk' and not a.script
    system = system_prompt(a.mode)
    prompt = user_prompt(task, a.mode, start, len(subset), a.max_numbers, a.max_turns, a.format_retries,
                         a.max_tokens, a.token_budget if sdk else None, warm=a.start_weights is not None,
                         start_score=start_score, turn_limit=a.max_turns if sdk else None)
    tools = tool_specs(a.mode, writable_names(start), a.tool_timeout, default_device())
    save(out / 'prompt.json', {'system': system, 'user': prompt, 'tools': tools})
    print(f'Prompt: {out / "prompt.json"}')
    if a.dry_run:
        return

    log = []
    episode = Episode(a.mode, start, images, labels, classes, out / 'workspace',
                      partial(apply_diff, max_numbers=a.max_numbers), a.device, a.tool_timeout, log)
    if sdk:
        from wsp.sdk_agent import SDKRun
        agent = SDKRun(episode, system, prompt, tools, a.model or 'claude-opus-5-5', out / 'agent_cwd', a.effort,
                       a.sdk_max_turns, a.max_tokens, max_replies=a.max_turns)
    elif a.script:
        agent = ScriptedAgent.from_file(a.script)
    else:
        agent = {'anthropic': AnthropicAgent, 'openai': OpenAIAgent, 'openrouter': OpenRouterAgent}[a.provider](
            system, prompt, tools, a.model or {'anthropic': 'claude-sonnet-5-5', 'openai': 'gpt-6.1-sol',
                                               'openrouter': 'anthropic/claude-opus-5.5'}[a.provider],
            a.max_tokens, a.effort)

    progress = {'logged': 0, 'time': time.monotonic(), 'pending': None, 'length': 0, 'replies': 0}

    def save_transcript():
        messages = agent.transcript()
        if progress['pending'] is not None and len(messages) == progress['length']:
            messages = messages + [pending_block(progress['pending'])]
        save(out / 'transcript.json', messages)

    def checkpoint(turn, counted=True, results=None):
        progress['pending'], progress['length'] = results, len(agent.transcript())
        save(out / 'calls.json', log)
        save_transcript()
        save(out / 'usage.json', agent.usage)
        now = time.monotonic()
        entries, progress['logged'] = log[progress['logged']:], len(log)
        print(turn_line(turn, entries, a.mode, episode.locked, now - progress['time'],
                        agent.usage[-1] if agent.usage else None, counted), flush=True)
        progress['time'] = now

    def sdk_reply(seconds, usage):
        progress['replies'] += 1
        save(out / 'calls.json', log)
        save_transcript()
        save(out / 'usage.json', agent.usage)
        entries, progress['logged'] = log[progress['logged']:], len(log)
        print(turn_line(progress['replies'], entries, a.mode, episode.locked, seconds, usage), flush=True)

    try:
        if sdk:
            agent.on_reply = sdk_reply
            asyncio.run(agent.run())
        else:
            run_episode(agent, episode, a.max_turns, checkpoint, a.format_retries)
    finally:
        save(out / 'calls.json', log)
        save_transcript()
        save(out / 'usage.json', agent.usage)
        if sdk and agent.result:
            save(out / 'sdk_result.json', agent.result)
            print(f'sdk result: {agent.result.get("subtype")}, {agent.result.get("num_turns")} turns, '
                  f'${agent.result.get("total_cost_usd")}', flush=True)
        if sdk and agent.turns['closed']:
            print(f'turn limit reached: {a.max_turns} replies without an accepted submit()', flush=True)
        problems = audit(a.mode, log, agent.content_types())
        save(out / 'audit.json', {'clean': not problems, 'problems': problems})
    if problems:
        sys.exit(f'AUDIT FAILED: {problems}')
    if episode.submission is None:
        save(out / 'result.json', {'error': 'no valid submission'})
        sys.exit('No valid submission')

    torch.save(episode.submission, out / 'weights.pt')
    result = {'calls': len(log)}
    for split in ['val'] + (['test'] if a.evaluate_test else []):
        held_out = datasets.load_split(task.dataset, split, a.data_dir)
        result[split] = {name: summarize(evaluate(w, held_out, classes, a.device))
                         for name, w in (('start', start), ('submitted', episode.submission))}
        print(f'{split}: start {result[split]["start"]["accuracy"]:.4f} -> '
              f'submitted {result[split]["submitted"]["accuracy"]:.4f}')
    save(out / 'result.json', result)
    print(f'Run directory: {out}')


if __name__ == '__main__':
    main()
