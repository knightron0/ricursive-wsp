#!/usr/bin/env python3
"""Recompute the README tables from results/. Run from the repository root: python3 code/analysis/summarize.py"""
import json
P = {x['id']: x for x in json.load(open('code/evalkit/prompts.json'))}


def split(r):
    if '<think>' not in r and '</think>' not in r: return 'no_think', r.strip()
    if '</think>' in r: return 'finished', r.split('</think>')[-1].strip()
    return 'unfinished', ''


def degenerate(r):
    w = r.split(); return len(w) >= 8 and len(set(w)) / len(w) < 0.5


def score(pid, replies, judge=None):
    chk = lambda a: bool(eval(P[pid]['check'], {"reply": a}))
    st = [split(r) for r in replies]
    ok = [s != 'unfinished' and bool(a) and chk(a) for s, a in st]
    out = dict(n=len(replies), shown=sum(ok), no_answer=sum(s == 'unfinished' for s, _ in st),
               garbage=sum(degenerate(a) for s, a in st if s != 'unfinished' and a))
    if judge is not None:
        out['correct'] = sum(x >= 0.5 for x in judge); out['both'] = sum(1 for o, x in zip(ok, judge) if o and x >= 0.5)
    return out


J = json.load(open('results/system_prompts/final_pirate_and_caps_v1_judged.json'))
print("== final (2,500-token completion, answer text only, 32 test questions)")
for k, v in J['rows'].items(): print('pirate   ', f'{k:24s}', score('pirate', v, J['judge'][k]))
for k, v in J['caps_rows'].items(): print('uppercase', f'{k:24s}', score('caps', v, J['judge']['caps/' + k]))
C = json.load(open('results/system_prompts/final_caps_v2_judged.json'))
print('uppercase', f'{"edit v2":24s}', score('caps', C['v2'], C['judge']))
E2 = json.load(open('results/system_prompts/eval_400tok_all_prompts.json'))
print("== 400-token run, answer text only")
for pid in ('terse', 'haiku', 'conch'):
    print(f'{pid:6s}', 'unedited', score(pid, E2['base']), '| prompted', score(pid, E2[pid + '/prompted']), '| edited', score(pid, E2[pid + '/edited']))
F = json.load(open('results/self_edit/fact_verify.json'))
print("== Ripple's fact edit")
for k in ('base', 'ripple_edit'): print(k, F[k]['probs'], F[k]['neigh'])
