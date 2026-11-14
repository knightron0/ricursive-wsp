"""Weight-space programming harness: an LLM handwrites ResNet-18 weights.

Tasks (see wsp/tasks.py):
  scratch-cifar10      build a CIFAR-10 ResNet-18 from random init
  cifar10-to-cifar100  turn a trained CIFAR-10 ResNet-18 into a CIFAR-100 one

Each round the LLM writes `build(weights, data, info) -> weights`. The program runs
in a restricted subprocess: no gradients, no network, and only `data`, a fixed
class-balanced X% subset of the target train split (--data-fraction). The result is
kept if validation accuracy improves. Test is used once, after freezing.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import random
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import torch

from wsp import data as datasets
from wsp.evaluation import evaluate, summarize, weight_stats
from wsp.model import ARCHITECTURE, default_device
from wsp.tasks import DATASET_DESCRIPTIONS, TASKS, start_weights, validate_weights
from wsp.toolkit import DOC as TOOLKIT_DOC

ROOT = Path(__file__).resolve().parent
EXAMPLES = {'scratch-cifar10': ('examples/programs/random_features.py', 0.276),
            'cifar10-to-cifar100': ('examples/programs/linear_probe.py', 0.315)}
OPENROUTER_URL = 'https://openrouter.ai/api/v1'
FIREWORKS_URL = 'https://api.fireworks.ai/inference/v1'
PROVIDER_KEYS = {'openai': 'OPENAI_API_KEY', 'openrouter': 'OPENROUTER_API_KEY', 'fireworks': 'FIREWORKS_API_KEY',
                 'modal': 'MODAL_RIPPLE_API_KEY'}
# 'modal': self-hosted Ripple-65B on Modal (modal_ripple.py); URL from `modal deploy`, ending in /v1
CHAT_PROVIDERS = {'openrouter': OPENROUTER_URL, 'fireworks': FIREWORKS_URL,
                  'modal': os.environ.get('MODAL_RIPPLE_URL')}

INSTRUCTIONS = f"""You are doing weight-space programming: you write the weights of a neural
network directly, as a Python program, instead of training it. The task, architecture,
your data budget and the current diagnostics are in the input.

Return a complete Python module that defines:

    def build(weights, data, info):
        ...
        return weights

- `weights`: dict of the CURRENT BEST state_dict tensors (CPU, float32 except the int64
  num_batches_tracked). Return a dict with exactly the same names and shapes, all finite.
- `data`: {{"images": uint8 [N,3,32,32], "labels": int64 [N], "classes": [...]}}: the only
  training data you may use (a fixed class-balanced subset of the train split).
- `info`: task, num_classes, data budget, and the current validation metrics.
- `torch`, `math` and `toolkit` are available. There are NO gradients (autograd and
  torch.optim are disabled), NO network access, and NO file access beyond `data`.
  Weights must be computed analytically: statistics, patches, PCA/ZCA whitening,
  clustering, least squares, BatchNorm statistics, hand-designed filters, etc.
- The program has a time limit (see info["time_limit_seconds"]), so keep it efficient.

{TOOLKIT_DOC}

Your program's output replaces the current best only if validation accuracy improves
(ties broken by lower loss); otherwise the current best is kept. Rounds build on each
other, so you can construct the network incrementally (e.g. layer by layer) or rewrite
it entirely. Learn from previous attempts and their errors. Return a short rationale
and the program. Do not claim results before evaluation.
"""

NO_DATA_INSTRUCTIONS = f"""You are doing weight-space programming: you write the weights of a neural
network directly, as a Python program, instead of training it. The task, architecture,
dataset description and the current diagnostics are in the input.

You get NO real training data. Everything must come from your own knowledge of the
dataset, natural images and convolutional networks. You get exactly ONE attempt.

Return a complete Python module that defines:

    def build(weights, data, info):
        ...
        return weights

- `weights`: dict of the starting state_dict tensors (CPU, float32 except the int64
  num_batches_tracked). Return a dict with exactly the same names and shapes, all finite.
- `data`: empty ({{"images": uint8 [0,3,32,32], "labels": int64 [0], "classes": [...]}}).
- `info`: task, num_classes, dataset description and validation metrics of the start weights.
- `torch`, `math` and `toolkit` are available. There are NO gradients (autograd and
  torch.optim are disabled), NO network access and NO file access.
- The program may run for up to info["time_limit_seconds"] seconds, with a GPU at
  toolkit.device. Use that compute; a short program wastes the budget.

You MAY synthesize your own images inside the program: procedurally render CIFAR-like
32x32 uint8 scenes from what you know about each class (backgrounds, colors, layout, shapes,
textures, typical framing), with whatever labels you assign. You can then run every toolkit
function on them exactly as on real data: BatchNorm statistics with capture(reduce="mean_var"),
filters from patches/whiten/kmeans, a ridge classifier head on pooled features. You may
also hand-design weights directly (color-opponent and oriented-edge/Gabor filters at several
scales, center-surround, identity residual paths that carry features to the head, spatial
pooling of layout cues such as sky above / ground below) or combine both approaches. The
classifier head is usually the bottleneck: calibrate it carefully (sensible logit scale, class
balance), since you are scored on validation accuracy, then loss.

{TOOLKIT_DOC}

Think carefully: there is no second round. Return a short rationale and the program.
Do not claim results before evaluation.
"""

PROPOSAL_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'rationale': {'type': 'string'}, 'program': {'type': 'string'}},
    'required': ['rationale', 'program'],
}


class ProgramError(Exception):
    """A proposed program failed; reported back to the agent, run continues."""


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_program(weights, program, info, data_path, timeout, allow_gradients=False):
    """Run `build` in a restricted subprocess on a copy of the weights; return validated weights."""
    with tempfile.TemporaryDirectory() as workdir:
        work = Path(workdir)
        torch.save(weights, work / 'input.pt')
        (work / 'program.py').write_text(program)
        save(work / 'info.json', info)
        env = {k: v for k, v in os.environ.items() if 'API_KEY' not in k.upper()}
        env['PYTHONPATH'] = str(ROOT)
        command = [sys.executable, '-m', 'wsp.apply', str(work), str(data_path)]
        if allow_gradients:
            command.append('--allow-gradients')
        try:
            process = subprocess.run(command, cwd=work, env=env, capture_output=True,
                                     text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise ProgramError(f'Program timed out after {timeout}s') from None
        if process.returncode:
            raise ProgramError(f'Program failed:\n{process.stderr[-3000:]}')
        try:
            candidate = torch.load(work / 'output.pt', weights_only=True)
        except Exception as error:
            raise ProgramError(f'Could not load returned weights: {error}') from None
    return validate_weights(weights, candidate, ProgramError)


def better(new, old):
    return (new['accuracy'], -new['loss']) > (old['accuracy'], -old['loss'])


def api_proposer(model, max_output_tokens, reasoning_effort=None, instructions=INSTRUCTIONS):
    from openai import OpenAI
    client = OpenAI(timeout=900, max_retries=0)

    def propose(payload):
        response = client.responses.create(
            model=model, instructions=instructions,
            input=json.dumps(payload, allow_nan=False), store=False,
            max_output_tokens=max_output_tokens,
            text={'format': {'type': 'json_schema', 'name': 'weight_program',
                             'strict': True, 'schema': PROPOSAL_SCHEMA}},
            **({'reasoning': {'effort': reasoning_effort}} if reasoning_effort else {}),
        )
        if response.status != 'completed' or not response.output_text:
            raise RuntimeError('API response incomplete or refused; no program applied')
        return json.loads(response.output_text), {
            'response_id': response.id,
            'usage': response.usage.model_dump() if response.usage else None,
        }
    return propose


CODE_REPLY_FORMAT = """
Reply format (plain text, NOT JSON): first a short rationale paragraph, then the complete
Python module in a single fenced block:

```python
def build(weights, data, info):
    ...
    return weights
```
"""


def _json_proposal(text):
    """Fuzzy JSON: tolerate surrounding prose/fences and raw newlines inside strings."""
    candidates = [text]
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get('program'), str):
            return {'rationale': str(value.get('rationale', '')), 'program': value['program']}
    return None


def parse_reply(text):
    """Extract {rationale, program} from a model reply: a ```python block, JSON, or bare code."""
    proposal = _parse_reply(text)
    proposal['program'] = proposal['program'].rstrip() + '\n'
    return proposal


def _parse_reply(text):
    text = text.strip()
    if text.startswith('{'):
        proposal = _json_proposal(text)
        if proposal:
            return proposal
    blocks = [(m.start(), m.group(1)) for m in
              re.finditer(r'```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)', text, re.S)]
    with_build = [b for b in blocks if 'def build' in b[1]]
    if with_build:
        position, code = max(with_build, key=lambda b: len(b[1]))
        return {'rationale': text[:blocks[0][0]].strip()[:3000], 'program': code}
    proposal = _json_proposal(text)
    if proposal:
        return proposal
    match = re.search(r'^(?:import |from |def build|"""|\'\'\')', text, re.M)
    if match and 'def build' in text:
        return {'rationale': text[:match.start()].strip()[:3000], 'program': text[match.start():]}
    raise ValueError('Reply contains no Python program with build()')


def openrouter_proposer(model, max_output_tokens, reasoning_effort=None, provider='openrouter',
                        response_format='code', instructions=INSTRUCTIONS):
    """Any model on an OpenAI-compatible Chat Completions API (OpenRouter, Fireworks).
    response_format='code': rationale + ```python block (robust for weaker models);
    'json': strict JSON schema."""
    from openai import OpenAI
    client = OpenAI(base_url=CHAT_PROVIDERS[provider], api_key=os.environ[PROVIDER_KEYS[provider]],
                    timeout=900, max_retries=0)
    name = provider.capitalize()
    extra = {}
    if response_format == 'json':
        system = instructions
        extra['response_format'] = {'type': 'json_schema', 'json_schema': {
            'name': 'weight_program', 'strict': True, 'schema': PROPOSAL_SCHEMA}}
    else:
        system = instructions + CODE_REPLY_FORMAT
    if reasoning_effort:
        extra['extra_body'] = {'reasoning': {'effort': reasoning_effort}}

    def propose(payload):
        response = client.chat.completions.create(
            model=model, max_tokens=max_output_tokens,
            messages=[{'role': 'system', 'content': system},
                      {'role': 'user', 'content': json.dumps(payload, allow_nan=False)}],
            **extra,
        )
        choice = response.choices[0] if response.choices else None
        if choice is None or choice.finish_reason not in ('stop', None) or not choice.message.content:
            reason = choice.finish_reason if choice else 'no choices'
            raise RuntimeError(f'{name} response incomplete ({reason}); no program applied')
        try:
            proposal = parse_reply(choice.message.content)
        except ValueError:
            raise RuntimeError(f'{name} reply contained no program; no program applied') from None
        return proposal, {
            'response_id': response.id, 'model': getattr(response, 'model', model),
            'raw_reply': choice.message.content,
            'usage': response.usage.model_dump() if response.usage else None,
        }
    return propose


def with_retries(propose, retries, base_delay=5.0, max_delay=120.0, sleep=time.sleep):
    """Retry rate limits (429), server errors (5xx), timeouts and connection errors with
    exponential backoff, honoring Retry-After when the provider sends it."""
    def call(payload):
        for attempt in range(retries + 1):
            try:
                return propose(payload)
            except Exception as error:
                status = getattr(error, 'status_code', None)
                transient = status == 429 or (status is not None and status >= 500) or \
                    type(error).__name__ in ('APIConnectionError', 'APITimeoutError', 'RateLimitError')
                if not transient or attempt == retries:
                    raise
                headers = getattr(getattr(error, 'response', None), 'headers', None) or {}
                delay = min(max_delay, base_delay * 2 ** attempt) * (0.75 + 0.5 * random.random())
                try:
                    delay = max(delay, float(headers.get('retry-after')))
                except (TypeError, ValueError):
                    pass
                log(f'API {status or type(error).__name__}: retrying in {delay:.0f}s '
                    f'(attempt {attempt + 1}/{retries})')
                sleep(delay)
    return call


def clean_program(source):
    """Repair JSON/markdown debris some models leak into the program string (a trailing `}"`,
    code fences). Only used when the original fails to compile and the cleaned version compiles."""
    def compiles(code):
        try:
            compile(code, 'program.py', 'exec')
            return True
        except (SyntaxError, ValueError):
            return False

    if compiles(source):
        return source, False
    code = source.strip()
    fence = re.match(r'^```[a-zA-Z]*\n(.*?)\n```\s*$', code, re.S)
    if fence:
        code = fence.group(1)
    lines = code.splitlines()
    while lines and re.fullmatch(r'[\s"\'}\]`,]*(#.*)?', lines[-1]):
        lines.pop()
    code = '\n'.join(lines) + '\n'
    return (code, True) if compiles(code) else (source, False)


def file_proposer(paths):
    """Offline proposer: replay hand-written programs in order (no API)."""
    programs = iter(paths)

    def propose(payload):
        path = next(programs)
        return {'rationale': f'Hand-written program {path}', 'program': Path(path).read_text()}, {}
    return propose


def search(start, evaluate_fn, propose, apply_fn, rounds, output, context, history_size=3, candidates=1, stagger=2.0):
    """Hill climb. Each round asks for `candidates` programs in parallel, runs each as soon as it
    arrives, and keeps the best one if it beats the current best. context: static task info."""
    log('Evaluating start weights on val...')
    best, metrics = start, evaluate_fn(start)
    log(f'Start val accuracy {metrics["accuracy"]:.4f}. Output: {output}')
    save(output / 'baseline_val.json', metrics)
    torch.save(best, output / 'best.pt')
    save(output / 'best_metrics.json', metrics)
    lineage, history = [], []
    save(output / 'lineage.json', lineage)
    for number in range(1, rounds + 1):
        info = {**context, 'metrics': metrics}
        payload = {**info, 'architecture': ARCHITECTURE, 'parameters': weight_stats(best),
                   'accepted_programs': [{k: e[k] for k in ('round', 'rationale', 'accuracy')} for e in lineage],
                   'previous_attempts': history[-history_size * candidates:]}
        directory = output / f'round_{number:03d}'
        directory.mkdir()
        save(directory / 'input.json', payload)

        def payload_for(k):
            if candidates == 1:
                return payload
            return {**payload, 'parallel_candidate': {
                'index': k, 'of': candidates,
                'note': f'{candidates} programs are being written independently for this round and the best '
                        'one is kept. Pursue a distinct idea rather than the most obvious one.'}}

        log(f'Round {number}/{rounds}: waiting for {candidates} program(s) from the model...')
        started = time.time()
        records, winner = [], None
        with ThreadPoolExecutor(max_workers=candidates) as pool:
            futures = {}
            for k in range(1, candidates + 1):
                if k > 1:
                    time.sleep(stagger)  # avoid bursting a provider's rate limit
                futures[pool.submit(propose, payload_for(k))] = k
            for future in as_completed(futures):
                k = futures[future]
                label = f'Round {number}' + (f'.{k}' if candidates > 1 else '')
                place = directory if candidates == 1 else directory / f'cand_{k}'
                place.mkdir(exist_ok=True)
                try:
                    proposal, metadata = future.result()
                    if not isinstance(proposal.get('program'), str) or not isinstance(proposal.get('rationale'), str):
                        raise RuntimeError('Proposal must contain string rationale and program')
                except Exception as error:
                    log(f'{label}: API call failed after retries ({error}); skipping this candidate')
                    save(place / 'result.json', {'round': number, 'candidate': k, 'api_error': str(error)})
                    continue
                proposal['program'], repaired = clean_program(proposal['program'])
                if repaired:
                    log(f'{label}: stripped stray JSON/markdown characters from the program')
                log(f'{label}: got program ({time.time() - started:.0f}s). '
                    f'Rationale: {proposal["rationale"][:300]}')
                (place / 'program.py').write_text(proposal['program'])
                record = {'round': number, 'candidate': k, 'rationale': proposal['rationale'],
                          'program': proposal['program'], 'program_repaired': repaired}
                try:
                    log(f'{label}: running program (no grads, no network)...')
                    run_started = time.time()
                    weights = apply_fn(best, proposal['program'], info)
                    log(f'{label}: program finished ({time.time() - run_started:.0f}s); evaluating on val...')
                    result = evaluate_fn(weights)
                except (ProgramError, ValueError) as error:
                    record.update(accepted=False, error=str(error))
                    log(f'{label}: ERROR ({str(error).strip().splitlines()[-1][:300]})')
                else:
                    record.update(accepted=False, metrics=summarize(result),
                                  delta_accuracy=result['accuracy'] - metrics['accuracy'])
                    log(f'{label}: val accuracy {result["accuracy"]:.4f} (current best {metrics["accuracy"]:.4f})')
                    if better(result, metrics) and (winner is None or better(result, winner[1])):
                        winner = (record, result, weights, place)
                records.append((record, metadata, place))
        if not records:
            reason = f'All API calls failed in round {number}; stopping early with the best weights so far'
            log(reason)
            save(output / 'stopped_early.json', {'round': number, 'reason': reason})
            break
        if winner:
            record, result, weights, place = winner
            record['accepted'] = True
            best, metrics = weights, result
            lineage.append({'round': number, 'candidate': record['candidate'], 'rationale': record['rationale'],
                            'program_file': str((place / 'program.py').relative_to(output)),
                            'accuracy': result['accuracy']})
            torch.save(best, output / 'best.pt')
            save(output / 'best_metrics.json', metrics)
            save(output / 'lineage.json', lineage)
        for record, metadata, place in records:
            history.append(record)
            save(place / 'result.json', {**record, **metadata})
        status = f'ACCEPTED candidate {winner[0]["candidate"]}' if winner and candidates > 1 else \
            'ACCEPTED' if winner else 'no improvement'
        log(f'Round {number}: {status}; best val accuracy {metrics["accuracy"]:.4f} '
            f'({time.time() - started:.0f}s)')
    return best, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', choices=sorted(TASKS), required=True)
    parser.add_argument('--data-fraction', type=float, required=True,
                        help='Fraction (0,1] of the target train split the program may use, e.g. 0.1')
    parser.add_argument('--data-seed', type=int, default=0, help='Seed for choosing the data subset')
    parser.add_argument('--source', type=Path, default=Path('checkpoints/resnet18_cifar10.pt'),
                        help='Trained CIFAR-10 checkpoint (cifar10-to-cifar100 only)')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--provider', choices=sorted(PROVIDER_KEYS), default='openai')
    parser.add_argument('--model', help='Model ID; default $OPENAI_MODEL or gpt-6-astra (openai), '
                        '$OPENROUTER_MODEL (openrouter), e.g. anthropic/claude-opus-5-5')
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--api-retries', type=int, default=8,
                        help='Retries per API call on rate limits / server errors, with backoff')
    parser.add_argument('--response-format', choices=['auto', 'json', 'code'], default='auto',
                        help='How the model returns its program: code = rationale + ```python block, '
                             'json = strict JSON schema; auto = json for openai, code otherwise')
    parser.add_argument('--candidates', type=int, default=1,
                        help='Programs requested in parallel per round; the best one is kept')
    parser.add_argument('--reasoning-effort', choices=['minimal', 'low', 'medium', 'high', 'xhigh'],
                        help='Model reasoning effort (lower is faster); default: provider default')
    parser.add_argument('--max-output-tokens', type=int, default=32000)
    parser.add_argument('--program-timeout', type=float, default=900)
    parser.add_argument('--history', type=int, default=3, help='Previous attempts shown to the model')
    parser.add_argument('--allow-gradients', action='store_true', help='Let programs use autograd/torch.optim')
    parser.add_argument('--no-example', action='store_true',
                        help='Do not show the task\'s working example program in the prompt')
    parser.add_argument('--device', default=default_device())
    parser.add_argument('--output', type=Path,
                        default=Path('runs') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    parser.add_argument('--programs', type=Path, nargs='+', help='Run hand-written programs instead of the API')
    parser.add_argument('--dry-run', action='store_true', help='Evaluate start weights, save the first prompt; no API')
    parser.add_argument('--evaluate-test', action='store_true', help='After freezing, score start and best on test once')
    args = parser.parse_args()
    if not 0 <= args.data_fraction <= 1:
        parser.error('--data-fraction must be in [0, 1] (0 = no training data)')
    if args.candidates < 1 or args.rounds < 1 or args.max_output_tokens < 1 or args.program_timeout <= 0 or args.history < 0:
        parser.error('Rounds, token and timeout limits must be positive')
    if args.dry_run and (args.evaluate_test or args.programs):
        parser.error('--dry-run cannot be combined with --evaluate-test or --programs')
    key = PROVIDER_KEYS[args.provider]
    if args.model is None:
        args.model = (os.environ.get('OPENAI_MODEL', 'gpt-6-astra') if args.provider == 'openai'
                      else os.environ.get(f'{args.provider.upper()}_MODEL',
                                          'ripple' if args.provider == 'modal' else None))
    using_api = not (args.dry_run or args.programs)
    if using_api and args.provider == 'modal' and not CHAT_PROVIDERS['modal']:
        parser.error('Set MODAL_RIPPLE_URL (https://<workspace>--ripple-serve.modal.run/v1)')
    if using_api and not os.environ.get(key):
        parser.error(f'Set {key}, or use --programs or --dry-run')
    if using_api and not args.model:
        parser.error(f'Pass --model (or set {args.provider.upper()}_MODEL)')
    task = TASKS[args.task]
    if task.needs_source and not args.source.exists():
        parser.error(f'{args.source} not found; run `python train_base.py` first')

    source = torch.load(args.source, weights_only=True) if task.needs_source else None
    start = start_weights(task, source)
    classes = datasets.class_names(task.dataset, args.data_dir)
    val = datasets.load_split(task.dataset, 'val', args.data_dir)
    train_images, train_labels = datasets.load_split(task.dataset, 'train', args.data_dir)
    no_data = args.data_fraction == 0
    subset = (torch.zeros(0, dtype=torch.long) if no_data
              else datasets.stratified_subset(train_labels, args.data_fraction, args.data_seed))
    evaluate_fn = lambda weights: evaluate(weights, val, classes, args.device)

    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'config.json', {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                       'programs': [str(p) for p in args.programs or []],
                                       'source_sha256': sha256(args.source) if task.needs_source else None})
    save(args.output / 'subset_indices.json', subset.tolist())
    data_path = args.output / 'data_subset.pt'
    torch.save({'images': train_images[subset].clone(), 'labels': train_labels[subset].clone(),
                'classes': classes}, data_path)
    per_class = torch.bincount(train_labels[subset], minlength=task.num_classes)
    del train_images, train_labels
    context = {'task': task.name, 'task_description': task.description, 'num_classes': task.num_classes,
               'classes': classes, 'gradients_allowed': args.allow_gradients,
               'time_limit_seconds': args.program_timeout,
               'dataset': DATASET_DESCRIPTIONS[task.dataset],
               'data_budget': {'fraction_of_train': args.data_fraction, 'examples': len(subset),
                               'per_class_min': per_class.min().item(), 'per_class_max': per_class.max().item()}}
    if no_data:
        context['data_budget']['note'] = (
            'NO TRAINING DATA: data["images"] and data["labels"] are empty. Every weight must come '
            'from your own knowledge of the dataset and of how CNNs work (hand-designed filters, '
            'analytically chosen BatchNorm parameters, a hand-built classifier head, ...). '
            'toolkit functions that need images will not help.')
    if not args.no_example and not no_data:
        path, score = EXAMPLES[task.name]
        context['example_program'] = {
            'note': 'A known-working reference program for this task that uses the toolkit correctly'
                    + (f' (about {score:.1%} val accuracy at 10% data)' if score else '')
                    + '. It is a simple baseline: start from it or replace it, but aim to do much better.',
            'source': (ROOT / path).read_text()}

    instructions = NO_DATA_INSTRUCTIONS if no_data else INSTRUCTIONS
    if args.dry_run:
        metrics = evaluate_fn(start)
        info = {**context, 'metrics': metrics}
        save(args.output / 'preview.json', {'instructions': instructions, 'schema': PROPOSAL_SCHEMA,
             'input': {**info, 'architecture': ARCHITECTURE, 'parameters': weight_stats(start),
                       'accepted_programs': [], 'previous_attempts': []}})
        print(f'Dry run: start val accuracy {metrics["accuracy"]:.4f}; {len(subset)} training examples '
              f'available; no API calls. Files: {args.output}')
        return

    if args.programs:
        propose, rounds = file_proposer(args.programs), len(args.programs)
    else:
        if args.provider == 'openai':
            base = api_proposer(args.model, args.max_output_tokens, args.reasoning_effort, instructions)
        else:
            fmt = 'code' if args.response_format == 'auto' else args.response_format
            base = openrouter_proposer(args.model, args.max_output_tokens, args.reasoning_effort,
                                       args.provider, fmt, instructions)
        propose = with_retries(base, args.api_retries)
        rounds = args.rounds
    apply_fn = lambda weights, program, info: run_program(
        weights, program, info, data_path.resolve(), args.program_timeout, args.allow_gradients)
    best, metrics = search(start, evaluate_fn, propose, apply_fn, rounds, args.output, context, args.history,
                          1 if args.programs else args.candidates)

    frozen = sha256(args.output / 'best.pt')
    save(args.output / 'frozen.json', {'best_sha256': frozen, 'val_accuracy': metrics['accuracy']})
    if args.evaluate_test:
        test = datasets.load_split(task.dataset, 'test', args.data_dir)
        start_test, best_test = (summarize(evaluate(w, test, classes, args.device)) for w in (start, best))
        save(args.output / 'final_test.json', {'start': start_test, 'best': best_test, 'best_sha256': frozen})
        print(f'Test accuracy: start {start_test["accuracy"]:.4f} -> best {best_test["accuracy"]:.4f}')
    print(f'Best weights: {args.output / "best.pt"}')


if __name__ == '__main__':
    main()
