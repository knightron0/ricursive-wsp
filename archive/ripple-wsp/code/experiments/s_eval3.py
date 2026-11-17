# Finish every reply: regenerate, with a 2500-token cap, each reply that did not reach an answer within 400 tokens.
EV3 = dict(EV)
def unfinished_idx(reps): return [i for i, r in enumerate(reps) if '<think>' in r and '</think>' not in r]
def finish(key, system, patch):
    idx = unfinished_idx(EV3[key])
    if not idx: return 0
    lab.rollback(); assert lab.n_changed() == 0
    if patch: lab.apply_patch('/work/patches/' + patch); assert lab.n_changed() > 0
    new = lab.gen([te[i] for i in idx], system, 2500, bs=8)
    for i, r in zip(idx, new): EV3[key][i] = r
    lab.rollback()
    return len(idx)
print('base', finish('base', None, None), flush=True)
for pid, patch in (('caps', 'caps_patch.pt'), ('pirate', 'pirate_s2.pt')):
    print(pid, 'prompted', finish(pid + '/prompted', P[pid]['system'], None), 'edited', finish(pid + '/edited', None, patch), flush=True)
    print(pid, '| base', score(pid, EV3['base']), '| prompted', score(pid, EV3[pid + '/prompted']), '| edited', score(pid, EV3[pid + '/edited']), flush=True)
json.dump(EV3, open('/work/eval3.json', 'w')); print("EVAL3 DONE")
