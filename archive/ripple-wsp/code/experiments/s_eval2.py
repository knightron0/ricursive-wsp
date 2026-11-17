# Corrected evaluation: long replies, score ONLY the final answer (text after the reasoning block).
import lab, wsp1, json, torch, logging, time, importlib
logging.disable(logging.WARNING)
if lab.model is None: lab.load()
Q = json.load(open('/work/queries.json')); P = {p['id']: p for p in json.load(open('/work/prompts.json'))}
MAXNEW = 400


def split_reply(r):
    """-> (state, answer). no_think: no reasoning block. finished: reasoning closed, answer follows. unfinished: cut off inside reasoning."""
    if '<think>' not in r and '</think>' not in r:
        return 'no_think', r.strip()
    if '</think>' in r:
        return 'finished', r.split('</think>')[-1].strip()
    return 'unfinished', ''


def score(pid, replies):
    chk = lambda a: bool(eval(P[pid]['check'], {"reply": a}))
    st = [split_reply(r) for r in replies]
    answered = [a for s, a in st if s != 'unfinished' and a]
    return dict(n=len(replies), think=sum(s != 'no_think' for s, _ in st), unfinished=sum(s == 'unfinished' for s, _ in st),
                empty=sum(s != 'unfinished' and not a for s, a in st), answered=len(answered), passed=sum(map(chk, answered)),
                degenerate=sum(map(wsp1.degenerate, answered)))


EV = globals().get('EV', {})
te = Q['test']
if 'base' not in EV:
    lab.rollback(); assert lab.n_changed() == 0
    EV['base'] = lab.gen(te, None, MAXNEW, bs=32); json.dump(EV, open('/work/eval2.json', 'w'))
    print('base generated', flush=True)
for pid, patch in (('caps', 'caps_patch.pt'), ('pirate', 'pirate_s2.pt'), ('haiku', 'haiku_patch.pt'), ('terse', 'terse_patch.pt'), ('conch', 'conch_patch.pt')):
    if pid + '/edited' in EV:
        continue
    lab.rollback(); assert lab.n_changed() == 0
    EV[pid + '/prompted'] = lab.gen(te, P[pid]['system'], MAXNEW, bs=32)
    lab.apply_patch('/work/patches/' + patch); npar = lab.n_changed(); assert npar > 0
    EV[pid + '/edited'] = lab.gen(te, None, MAXNEW, bs=32); EV[pid + '/params'] = npar
    lab.rollback()
    json.dump(EV, open('/work/eval2.json', 'w'))
    print(pid, 'params', npar, '| base', score(pid, EV['base']), '| prompted', score(pid, EV[pid + '/prompted']), '| edited', score(pid, EV[pid + '/edited']), flush=True)
print("EVAL2 DONE")
