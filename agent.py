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
import tempfile

import torch

from wsp import data as datasets
from wsp.evaluation import evaluate, summarize, weight_stats
from wsp.model import ARCHITECTURE, default_device
from wsp.tasks import TASKS, start_weights, validate_weights
from wsp.toolkit import DOC as TOOLKIT_DOC

ROOT = Path(__file__).resolve().parent
OPENROUTER_URL = 'https://openrouter.ai/api/v1'
PROVIDER_KEYS = {'openai': 'OPENAI_API_KEY', 'openrouter': 'OPENROUTER_API_KEY'}

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

PROPOSAL_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'rationale': {'type': 'string'}, 'program': {'type': 'string'}},
    'required': ['rationale', 'program'],
}


class ProgramError(Exception):
    """A proposed program failed; reported back to the agent, run continues."""


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


def api_proposer(model, max_output_tokens):
    from openai import OpenAI
    client = OpenAI(timeout=900, max_retries=0)

    def propose(payload):
        response = client.responses.create(
            model=model, instructions=INSTRUCTIONS,
            input=json.dumps(payload, allow_nan=False), store=False,
            max_output_tokens=max_output_tokens,
            text={'format': {'type': 'json_schema', 'name': 'weight_program',
                             'strict': True, 'schema': PROPOSAL_SCHEMA}},
        )
        if response.status != 'completed' or not response.output_text:
            raise RuntimeError('API response incomplete or refused; no program applied')
        return json.loads(response.output_text), {
            'response_id': response.id,
            'usage': response.usage.model_dump() if response.usage else None,
        }
    return propose


def parse_proposal(text):
    """Parse a JSON proposal, tolerating code fences or prose around it."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find('{'), text.rfind('}')
        if start == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def openrouter_proposer(model, max_output_tokens):
    """Any OpenRouter model via its OpenAI-compatible Chat Completions API."""
    from openai import OpenAI
    client = OpenAI(base_url=OPENROUTER_URL, api_key=os.environ['OPENROUTER_API_KEY'],
                    timeout=900, max_retries=0)

    def propose(payload):
        response = client.chat.completions.create(
            model=model, max_tokens=max_output_tokens,
            messages=[{'role': 'system', 'content': INSTRUCTIONS},
                      {'role': 'user', 'content': json.dumps(payload, allow_nan=False)}],
            response_format={'type': 'json_schema', 'json_schema': {
                'name': 'weight_program', 'strict': True, 'schema': PROPOSAL_SCHEMA}},
        )
        choice = response.choices[0] if response.choices else None
        if choice is None or choice.finish_reason not in ('stop', None) or not choice.message.content:
            reason = choice.finish_reason if choice else 'no choices'
            raise RuntimeError(f'OpenRouter response incomplete ({reason}); no program applied')
        try:
            proposal = parse_proposal(choice.message.content)
        except json.JSONDecodeError:
            raise RuntimeError('OpenRouter model did not return JSON; no program applied') from None
        return proposal, {
            'response_id': response.id, 'model': getattr(response, 'model', model),
            'usage': response.usage.model_dump() if response.usage else None,
        }
    return propose


def file_proposer(paths):
    """Offline proposer: replay hand-written programs in order (no API)."""
    programs = iter(paths)

    def propose(payload):
        path = next(programs)
        return {'rationale': f'Hand-written program {path}', 'program': Path(path).read_text()}, {}
    return propose


def search(start, evaluate_fn, propose, apply_fn, rounds, output, context, history_size=3):
    """context: static task info shown to the LLM and passed to programs."""
    best, metrics = start, evaluate_fn(start)
    save(output / 'baseline_val.json', metrics)
    torch.save(best, output / 'best.pt')
    save(output / 'best_metrics.json', metrics)
    lineage, history = [], []
    save(output / 'lineage.json', lineage)
    for number in range(1, rounds + 1):
        info = {**context, 'metrics': metrics}
        payload = {**info, 'architecture': ARCHITECTURE, 'parameters': weight_stats(best),
                   'accepted_programs': [{k: e[k] for k in ('round', 'rationale', 'accuracy')} for e in lineage],
                   'previous_attempts': history[-history_size:]}
        directory = output / f'round_{number:03d}'
        directory.mkdir()
        save(directory / 'input.json', payload)
        proposal, metadata = propose(payload)
        if not isinstance(proposal.get('program'), str) or not isinstance(proposal.get('rationale'), str):
            raise RuntimeError('Proposal must contain string rationale and program')
        (directory / 'program.py').write_text(proposal['program'])
        record = {'round': number, 'rationale': proposal['rationale'], 'program': proposal['program']}
        try:
            candidate = apply_fn(best, proposal['program'], info)
            result = evaluate_fn(candidate)
        except (ProgramError, ValueError) as error:
            record.update(accepted=False, error=str(error))
        else:
            accepted = better(result, metrics)
            record.update(accepted=accepted, metrics=summarize(result),
                          delta_accuracy=result['accuracy'] - metrics['accuracy'])
            if accepted:
                best, metrics = candidate, result
                lineage.append({'round': number, 'rationale': proposal['rationale'],
                                'program_file': f'{directory.name}/program.py', 'accuracy': result['accuracy']})
                torch.save(best, output / 'best.pt')
                save(output / 'best_metrics.json', metrics)
                save(output / 'lineage.json', lineage)
        history.append(record)
        save(directory / 'result.json', {**record, **metadata})
        status = 'accepted' if record['accepted'] else ('error' if 'error' in record else 'rejected')
        print(f'Round {number}: {status}; best val accuracy {metrics["accuracy"]:.4f}', flush=True)
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
    parser.add_argument('--max-output-tokens', type=int, default=32000)
    parser.add_argument('--program-timeout', type=float, default=900)
    parser.add_argument('--history', type=int, default=3, help='Previous attempts shown to the model')
    parser.add_argument('--allow-gradients', action='store_true', help='Let programs use autograd/torch.optim')
    parser.add_argument('--device', default=default_device())
    parser.add_argument('--output', type=Path,
                        default=Path('runs') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    parser.add_argument('--programs', type=Path, nargs='+', help='Run hand-written programs instead of the API')
    parser.add_argument('--dry-run', action='store_true', help='Evaluate start weights, save the first prompt; no API')
    parser.add_argument('--evaluate-test', action='store_true', help='After freezing, score start and best on test once')
    args = parser.parse_args()
    if not 0 < args.data_fraction <= 1:
        parser.error('--data-fraction must be in (0, 1]')
    if args.rounds < 1 or args.max_output_tokens < 1 or args.program_timeout <= 0 or args.history < 0:
        parser.error('Rounds, token and timeout limits must be positive')
    if args.dry_run and (args.evaluate_test or args.programs):
        parser.error('--dry-run cannot be combined with --evaluate-test or --programs')
    key = PROVIDER_KEYS[args.provider]
    if args.model is None:
        args.model = (os.environ.get('OPENAI_MODEL', 'gpt-6-astra') if args.provider == 'openai'
                      else os.environ.get('OPENROUTER_MODEL'))
    using_api = not (args.dry_run or args.programs)
    if using_api and not os.environ.get(key):
        parser.error(f'Set {key}, or use --programs or --dry-run')
    if using_api and not args.model:
        parser.error('Pass --model (or set OPENROUTER_MODEL), e.g. --model anthropic/claude-opus-5-5')
    task = TASKS[args.task]
    if task.needs_source and not args.source.exists():
        parser.error(f'{args.source} not found; run `python train_base.py` first')

    source = torch.load(args.source, weights_only=True) if task.needs_source else None
    start = start_weights(task, source)
    classes = datasets.class_names(task.dataset, args.data_dir)
    val = datasets.load_split(task.dataset, 'val', args.data_dir)
    train_images, train_labels = datasets.load_split(task.dataset, 'train', args.data_dir)
    subset = datasets.stratified_subset(train_labels, args.data_fraction, args.data_seed)
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
               'data_budget': {'fraction_of_train': args.data_fraction, 'examples': len(subset),
                               'per_class_min': per_class.min().item(), 'per_class_max': per_class.max().item()}}

    if args.dry_run:
        metrics = evaluate_fn(start)
        info = {**context, 'metrics': metrics}
        save(args.output / 'preview.json', {'instructions': INSTRUCTIONS, 'schema': PROPOSAL_SCHEMA,
             'input': {**info, 'architecture': ARCHITECTURE, 'parameters': weight_stats(start),
                       'accepted_programs': [], 'previous_attempts': []}})
        print(f'Dry run: start val accuracy {metrics["accuracy"]:.4f}; {len(subset)} training examples '
              f'available; no API calls. Files: {args.output}')
        return

    if args.programs:
        propose, rounds = file_proposer(args.programs), len(args.programs)
    else:
        make = api_proposer if args.provider == 'openai' else openrouter_proposer
        propose, rounds = make(args.model, args.max_output_tokens), args.rounds
    apply_fn = lambda weights, program, info: run_program(
        weights, program, info, data_path.resolve(), args.program_timeout, args.allow_gradients)
    best, metrics = search(start, evaluate_fn, propose, apply_fn, rounds, args.output, context, args.history)

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
