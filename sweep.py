"""Queue runs of `python -m wsp` over a grid, N at a time; finished runs are skipped, so re-running resumes.

    python sweep.py --name sol --provider openai --effort medium      # run the grid
    python sweep.py --name sol --collect                               # just rebuild results.csv
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent


def grid(tasks, fractions, seeds):
    for task in tasks:
        for seed in seeds:
            yield task, 'blind', 0.0, seed
            for mode in ('reasoned', 'full'):
                for f in fractions:
                    yield task, mode, f, seed


def run(cmd, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.parent / f'{out.name}.log', 'w') as log:
        return out, subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode


def collect(base):
    rows = []
    for cfg in sorted(base.glob('*/*/config.json')):
        c, d = json.loads(cfg.read_text()), cfg.parent
        r = json.loads((d / 'result.json').read_text()) if (d / 'result.json').exists() else {}
        audit = json.loads((d / 'audit.json').read_text()) if (d / 'audit.json').exists() else {}
        calls = json.loads((d / 'calls.json').read_text()) if (d / 'calls.json').exists() else []
        rows.append({'task': c['task'], 'mode': c['mode'], 'fraction': c['data_fraction'], 'seed': c['data_seed'],
                     'model': c.get('model') or c.get('provider'), 'val': r.get('val', {}).get('submitted', {}).get('accuracy'),
                     'test': r.get('test', {}).get('submitted', {}).get('accuracy'),
                     'start_test': r.get('test', {}).get('start', {}).get('accuracy'),
                     'calls': len(calls), 'audit_clean': audit.get('clean'), 'error': r.get('error'), 'dir': str(d)})
    if rows:
        with open(base / 'results.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--tasks', nargs='+', default=['scratch-cifar10'])
    p.add_argument('--fractions', type=float, nargs='+', default=[0.01, 0.1, 0.5])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--provider', default='openai')
    p.add_argument('--model')
    p.add_argument('--effort')
    p.add_argument('--jobs', type=int, default=3)
    p.add_argument('--collect', action='store_true')
    a = p.parse_args()
    base = ROOT / 'runs' / a.name
    if not a.collect:
        jobs = []
        for task, mode, f, seed in grid(a.tasks, a.fractions, a.seeds):
            out = base / task / f'{mode}_f{f:g}_s{seed}'
            if (out / 'result.json').exists() and 'error' not in json.loads((out / 'result.json').read_text()):
                continue
            if out.exists():  # unfinished earlier attempt: keep it, out of the way
                out.rename(out.with_name(f'{out.name}.failed{len(list(out.parent.glob(out.name + ".failed*")))}'))
            cmd = [sys.executable, '-m', 'wsp', '--task', task, '--mode', mode, '--data-fraction', str(f),
                   '--data-seed', str(seed), '--provider', a.provider, '--evaluate-test', '--output', str(out)]
            cmd += (['--model', a.model] if a.model else []) + (['--effort', a.effort] if a.effort else [])
            jobs.append((cmd, out))
        print(f'{len(jobs)} runs -> {base} ({a.jobs} at a time)', flush=True)
        with ThreadPoolExecutor(a.jobs) as pool:
            for fut in as_completed([pool.submit(run, cmd, out) for cmd, out in jobs]):
                out, code = fut.result()
                print(f'{out.relative_to(base)}: {"ok" if code == 0 else f"FAILED ({code}), see {out}.log"}', flush=True)
    for r in collect(base):
        print(f"{r['mode']:9} f={r['fraction']:<5g} s={r['seed']} test={r['test']} audit={r['audit_clean']} "
              f"calls={r['calls']} {r['error'] or ''}")


if __name__ == '__main__':
    main()
