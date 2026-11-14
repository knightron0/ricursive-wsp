"""Weight-space programming: an agent writes a literal diff over an initialized ResNet-18.

    python -m wsp --task scratch-cifar10 --mode blind
    python -m wsp --task scratch-cifar10 --mode reasoned --data-fraction 0.1
    python -m wsp --task scratch-cifar10 --mode full     --data-fraction 0.1 --evaluate-test
    python -m wsp ... --dry-run                      # write prompt + tool definitions, no API call
    python -m wsp ... --script calls.json            # replay scripted calls, no API call

Modes (see wsp/episode.py): blind = tools() only, no data; reasoned = phase 1 tools/data(pool),
then end_tools() permanently disables tools() and opens labels/evaluate/submit; full = everything. Val (and test, with
--evaluate-test) are scored once on the submitted diff and never shown to the agent.
"""
import argparse
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
from wsp.prompt import SYSTEM, tool_specs, user_prompt
from wsp.tasks import TASKS, apply_diff, start_weights, writable_names


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')


def describe(entry):
    first = entry['result'].split('\n', 1)[0]
    if entry.get('status') in ('refused', 'failed'):
        return f'{entry["call"]} ({entry["status"]}: {first.removeprefix(entry["call"] + "() failed: ")[:160]})'
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


def turn_line(turn, entries, mode, locked, seconds, usage):
    head = f'turn {turn}'
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
    p.add_argument('--labeled-share', type=float, default=0.5,
                   help='reasoned: share of the subset in the labeled set (the rest is the unlabeled pool)')
    p.add_argument('--provider', choices=['anthropic', 'openai', 'openrouter'], default='anthropic')
    p.add_argument('--model', help='default: claude-sonnet-5-5 (anthropic), gpt-6.1-sol (openai), anthropic/claude-opus-5.5 (openrouter)')
    p.add_argument('--effort', choices=['low', 'medium', 'high', 'xhigh', 'max'])
    p.add_argument('--max-tokens', type=int, default=64000, help='Per model turn')
    p.add_argument('--max-turns', type=int, default=40)
    p.add_argument('--max-numbers', type=int, default=20000, help='Most numbers a diff may hold')
    p.add_argument('--tool-timeout', type=float, default=300)
    p.add_argument('--script', type=Path, help='Replay scripted calls instead of calling the API')
    p.add_argument('--source', type=Path, default=Path('checkpoints/resnet18_cifar10.pt'))
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    p.add_argument('--device', default=default_device())
    p.add_argument('--output', type=Path,
                   default=Path('runs') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--evaluate-test', action='store_true')
    a = p.parse_args(argv)
    if (a.mode == 'blind') != (a.data_fraction == 0) or not 0 <= a.data_fraction <= 1:
        p.error('blind takes --data-fraction 0; reasoned/full need --data-fraction in (0, 1]')
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
    start = start_weights(task, torch.load(a.source, weights_only=True) if task.needs_source else None)
    classes = datasets.class_names(task.dataset, a.data_dir)
    images, labels = datasets.load_split(task.dataset, 'train', a.data_dir)
    subset = (torch.zeros(0, dtype=torch.long) if a.mode == 'blind'
              else datasets.stratified_subset(labels, a.data_fraction, a.data_seed))
    pool = 0
    if a.mode == 'reasoned':  # pool first, then the labeled set; each in original index order
        labeled_part = datasets.stratified_subset(labels[subset], a.labeled_share, a.data_seed + 1)
        is_labeled = torch.zeros(len(subset), dtype=torch.bool)
        is_labeled[labeled_part] = True
        subset = torch.cat([subset[~is_labeled], subset[is_labeled]])
        pool = int((~is_labeled).sum())
    images, labels = images[subset].clone(), labels[subset].clone()

    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'config.json', vars(a))
    save(out / 'subset_indices.json', {'indices': subset.tolist(), 'pool': pool})
    prompt, tools = user_prompt(task, a.mode, start, len(subset), a.max_numbers, pool), tool_specs(a.mode, writable_names(start))
    save(out / 'prompt.json', {'system': SYSTEM, 'user': prompt, 'tools': tools})
    print(f'Prompt: {out / "prompt.json"}')
    if a.dry_run:
        return

    log = []
    episode = Episode(a.mode, start, images, labels, classes, out / 'workspace',
                      partial(apply_diff, max_numbers=a.max_numbers), a.device, a.tool_timeout, log, pool)
    agent = (ScriptedAgent.from_file(a.script) if a.script
             else {'anthropic': AnthropicAgent, 'openai': OpenAIAgent, 'openrouter': OpenRouterAgent}[a.provider](
                 SYSTEM, prompt, tools, a.model or {'anthropic': 'claude-sonnet-5-5', 'openai': 'gpt-6.1-sol',
                                     'openrouter': 'anthropic/claude-opus-5.5'}[a.provider],
                 a.max_tokens, a.effort))

    progress = {'logged': 0, 'time': time.monotonic()}

    def checkpoint(turn):
        save(out / 'calls.json', log)
        save(out / 'transcript.json', agent.transcript())
        save(out / 'usage.json', agent.usage)
        now = time.monotonic()
        entries, progress['logged'] = log[progress['logged']:], len(log)
        print(turn_line(turn, entries, a.mode, episode.locked, now - progress['time'],
                        agent.usage[-1] if agent.usage else None), flush=True)
        progress['time'] = now

    try:
        run_episode(agent, episode, a.max_turns, checkpoint)
    finally:
        save(out / 'calls.json', log)
        save(out / 'transcript.json', agent.transcript())
        save(out / 'usage.json', agent.usage)
        problems = audit(a.mode, log, agent.content_types(), pool)
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
