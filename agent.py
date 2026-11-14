"""A minimal propose -> evaluate -> keep-the-best agent harness."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

INSTRUCTIONS = """Improve the supplied task by proposing explicit numeric parameter edits.
Use the current best parameters, evaluator feedback, and recent attempts.
Return a short rationale and edits. Values are absolute replacements, not deltas.
Omitted parameters stay unchanged. Higher evaluator scores are better.
Do not generate code or claim improvements before evaluation. There are no tools,
optimizers, or external data available to you; reason about the supplied evidence.
"""


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_task(task):
    if not isinstance(task.get('objective'), str) or not task['objective'].strip():
        raise ValueError('Task needs a nonempty objective')
    parameters = task.get('parameters')
    if not isinstance(parameters, dict) or not parameters or not all(finite(v) for v in parameters.values()):
        raise ValueError('Task parameters must be a nonempty map of names to finite numbers')
    for name in ['evaluator', 'final_evaluator']:
        if name in task and (not isinstance(task[name], list) or not task[name]
                             or not all(isinstance(arg, str) for arg in task[name])):
            raise ValueError(f'{name} must be an argv array')
    if 'evaluator' not in task:
        raise ValueError('Task needs an evaluator command')


def proposal_schema(parameters):
    return {
        'type': 'object', 'additionalProperties': False,
        'properties': {
            'rationale': {'type': 'string'},
            'edits': {'type': 'array', 'maxItems': len(parameters), 'items': {
                'type': 'object', 'additionalProperties': False,
                'properties': {'parameter': {'type': 'string', 'enum': list(parameters)},
                               'value': {'type': 'number'}},
                'required': ['parameter', 'value'],
            }},
        }, 'required': ['rationale', 'edits'],
    }


def apply_proposal(best, proposal):
    if not isinstance(proposal, dict) or not isinstance(proposal.get('rationale'), str) or not isinstance(proposal.get('edits'), list):
        raise ValueError('Expected rationale and edits')
    result, seen = best.copy(), set()
    for edit in proposal['edits']:
        name, value = edit['parameter'], edit['value']
        if name not in best or name in seen or not finite(value):
            raise ValueError('Unknown/duplicate parameter or non-finite value')
        seen.add(name)
        result[name] = value
    return result


def command_evaluator(command, cwd, timeout):
    def evaluate(parameters):
        env = os.environ.copy()
        env.pop('OPENAI_API_KEY', None)
        process = subprocess.run(command, cwd=cwd, env=env, shell=False,
                                 input=json.dumps(parameters, allow_nan=False),
                                 text=True, capture_output=True, timeout=timeout)
        if process.returncode:
            raise RuntimeError(f'Evaluator exited with status {process.returncode}: {process.stderr[-1000:]}')
        result = json.loads(process.stdout)
        if not isinstance(result, dict) or not finite(result.get('score')):
            raise ValueError('Evaluator must return a JSON object with a finite numeric score')
        json.dumps(result, allow_nan=False)
        return result
    return evaluate


def api_proposer(model, max_output_tokens, schema):
    from openai import OpenAI
    client = OpenAI(timeout=120, max_retries=0)

    def propose(payload):
        response = client.responses.create(
            model=model, instructions=INSTRUCTIONS,
            input=json.dumps(payload, allow_nan=False), store=False,
            max_output_tokens=max_output_tokens,
            text={'format': {'type': 'json_schema', 'name': 'parameter_proposal',
                             'strict': True, 'schema': schema}},
        )
        if response.status != 'completed' or not response.output_text:
            raise RuntimeError('API response incomplete or refused; no proposal applied')
        return json.loads(response.output_text), {
            'response_id': response.id,
            'usage': response.usage.model_dump() if response.usage else None,
        }
    return propose


def search(task, evaluate, propose, rounds, output):
    best = task['parameters'].copy()
    metrics = evaluate(best)
    save(output / 'baseline.json', metrics)
    save(output / 'best_parameters.json', best)
    save(output / 'best_metrics.json', metrics)
    history = []
    for number in range(1, rounds + 1):
        payload = {'objective': task['objective'], 'context': task.get('context', {}),
                   'current_best': best, 'current_metrics': metrics,
                   'previous_attempts': history[-3:]}
        save(output / f'round_{number:03d}_input.json', payload)
        proposal, metadata = propose(payload)
        record = {'round': number, 'proposal': proposal, **metadata}
        try:
            candidate = apply_proposal(best, proposal)
        except (ValueError, KeyError, TypeError) as error:
            # Keep invalid non-JSON numbers from breaking the log itself.
            record = {'round': number, 'accepted': False, 'error': str(error),
                      'invalid_proposal': repr(proposal), **metadata}
        else:
            result = evaluate(candidate)
            accepted = result['score'] > metrics['score']
            record.update(parameters=candidate, metrics=result, accepted=accepted)
            if accepted:
                best, metrics = candidate, result
                save(output / 'best_parameters.json', best)
                save(output / 'best_metrics.json', metrics)
        history.append(record)
        save(output / f'round_{number:03d}.json', record)
        print(f'Round {number}: {"accepted" if record["accepted"] else "rejected"}; best score {metrics["score"]:.6g}', flush=True)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--model', default=os.environ.get('OPENAI_MODEL', 'gpt-6-astra'))
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--max-output-tokens', type=int, default=6000)
    parser.add_argument('--eval-timeout', type=float, default=60)
    parser.add_argument('--output', type=Path, default=Path('runs') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    parser.add_argument('--dry-run', action='store_true', help='Evaluate baseline and preview prompt; no API calls')
    parser.add_argument('--evaluate-final', action='store_true', help='Run final_evaluator only after freezing parameters')
    args = parser.parse_args()
    if args.rounds < 1 or args.max_output_tokens < 1 or not finite(args.eval_timeout) or args.eval_timeout <= 0:
        parser.error('Round, token and timeout limits must be positive')
    if args.dry_run and args.evaluate_final:
        parser.error('--dry-run does not run the final evaluator')
    if not args.dry_run and not os.environ.get('OPENAI_API_KEY'):
        parser.error('Set OPENAI_API_KEY or use --dry-run')
    task = json.loads(args.task.read_text())
    validate_task(task)
    if args.evaluate_final and 'final_evaluator' not in task:
        parser.error('--evaluate-final requires final_evaluator in the task')
    schema = proposal_schema(task['parameters'])
    evaluate = command_evaluator(task['evaluator'], args.task.resolve().parent, args.eval_timeout)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'task.json', task)
    save(args.output / 'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    if args.dry_run:
        metrics = evaluate(task['parameters'])
        save(args.output / 'preview.json', {'instructions': INSTRUCTIONS, 'schema': schema,
             'input': {'objective': task['objective'], 'context': task.get('context', {}),
                       'current_best': task['parameters'], 'current_metrics': metrics, 'previous_attempts': []}})
        print(f'Dry run: baseline score {metrics["score"]:.6g}; no API calls. Files: {args.output}')
        return
    best = search(task, evaluate, api_proposer(args.model, args.max_output_tokens, schema), args.rounds, args.output)
    frozen_hash = hashlib.sha256((args.output / 'best_parameters.json').read_bytes()).hexdigest()
    save(args.output / 'frozen.json', {'parameters_sha256': frozen_hash})
    if args.evaluate_final:
        final = command_evaluator(task['final_evaluator'], args.task.resolve().parent, args.eval_timeout)
        save(args.output / 'final_results.json', {'result': final(best), 'parameters_sha256': frozen_hash})
    print(f'Best parameters: {args.output / "best_parameters.json"}')


if __name__ == '__main__':
    main()
