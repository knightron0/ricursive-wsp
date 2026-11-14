"""Data-efficiency sweep: run agent.py over a grid of data fractions x seeds, then collect results.

    python sweep.py --tasks cifar10-to-cifar100 scratch-cifar10            # LLM runs (default: Sol, 1 shot)
    python sweep.py --baseline                                             # no-LLM reference curve (example programs)
    python sweep.py --collect                                              # just rebuild results.csv

Each cell writes runs/sweep/<name>/<task>/f<frac>_s<seed>/; finished cells (final_test.json) are skipped,
so re-running resumes. Keys missing from the environment are read from .env.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from agent import EXAMPLES  # noqa: E402


def load_dotenv(path=ROOT / '.env'):
    if path.exists():
        for line in path.read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))


def cell_dir(base, task, frac, seed):
    return base / task / f'f{frac:g}_s{seed}'


def run_cell(cmd, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.parent / f'{out.name}.log'
    with open(log, 'w') as f:
        code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    return out, code, log


def collect(base):
    rows = []
    for cfg in sorted(base.glob('*/*/config.json')):
        d = cfg.parent
        c = json.loads(cfg.read_text())
        test = json.loads((d / 'final_test.json').read_text()) if (d / 'final_test.json').exists() else None
        val = json.loads((d / 'best_metrics.json').read_text()) if (d / 'best_metrics.json').exists() else {}
        base_val = json.loads((d / 'baseline_val.json').read_text()) if (d / 'baseline_val.json').exists() else {}
        accepted = len(json.loads((d / 'lineage.json').read_text())) if (d / 'lineage.json').exists() else None
        rows.append({'task': c['task'], 'data_fraction': c['data_fraction'], 'data_seed': c['data_seed'],
                     'model': 'baseline' if c.get('programs') else c['model'], 'rounds': c['rounds'],
                     'start_val': base_val.get('accuracy'), 'best_val': val.get('accuracy'),
                     'start_test': test and test['start']['accuracy'],
                     'best_test': test and test['best']['accuracy'],
                     'accepted_programs': accepted, 'finished': test is not None, 'dir': str(d.relative_to(ROOT))})
    out = base / 'results.csv'
    if rows:
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tasks', nargs='+', default=['cifar10-to-cifar100'])
    p.add_argument('--fractions', type=float, nargs='+', default=[0.05, 0.1, 0.25, 0.5, 1.0])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--provider', default='openai')
    p.add_argument('--model', default='gpt-6.1-sol')
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--program-timeout', type=float, default=900)
    p.add_argument('--jobs', type=int, default=3, help='Concurrent runs (they share one GPU for programs + eval)')
    p.add_argument('--baseline', action='store_true', help='Run the example program instead of the LLM')
    p.add_argument('--name', help='Sweep name (default: model, or "baseline")')
    p.add_argument('--collect', action='store_true', help='Only rebuild results.csv')
    p.add_argument('--extra', nargs=argparse.REMAINDER, default=[], help='Extra args passed to agent.py')
    a = p.parse_args()
    load_dotenv()
    base = ROOT / 'runs' / 'sweep' / (a.name or ('baseline' if a.baseline else a.model.replace('/', '_')))

    if not a.collect:
        jobs = []
        for task in a.tasks:
            for frac in a.fractions:
                for seed in a.seeds:
                    out = cell_dir(base, task, frac, seed)
                    if (out / 'final_test.json').exists():
                        continue
                    if out.exists():  # unfinished previous attempt: agent.py refuses existing dirs
                        out.rename(out.with_name(out.name + f'.failed{len(list(out.parent.glob(out.name + ".failed*")))}'))
                    cmd = [sys.executable, 'agent.py', '--task', task, '--data-fraction', str(frac),
                           '--data-seed', str(seed), '--program-timeout', str(a.program_timeout),
                           '--evaluate-test', '--output', str(out)]
                    if a.baseline:
                        cmd += ['--programs', EXAMPLES[task][0]]
                    else:
                        cmd += ['--provider', a.provider, '--model', a.model, '--rounds', str(a.rounds)]
                    jobs.append((cmd + a.extra, out))
        print(f'{len(jobs)} runs -> {base} ({a.jobs} at a time)', flush=True)
        with ThreadPoolExecutor(a.jobs) as pool:
            for fut in as_completed([pool.submit(run_cell, cmd, out) for cmd, out in jobs]):
                out, code, log = fut.result()
                status = 'ok' if code == 0 and (out / 'final_test.json').exists() else f'FAILED ({code}), see {log}'
                print(f'{out.relative_to(base)}: {status}', flush=True)

    rows, path = collect(base)
    for r in rows:
        print(f"{r['task']:22} f={r['data_fraction']:<5g} s={r['data_seed']} test {r['start_test']} -> {r['best_test']}"
              f"  accepted={r['accepted_programs']}")
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
