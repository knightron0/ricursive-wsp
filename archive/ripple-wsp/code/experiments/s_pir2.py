# Pirate edit v2: same fix as uppercase v2 (complete replies, answer tokens only, layers 34-78), strength chosen on VAL by "pirate AND correct".
SYSP = P['pirate']['system']; CAP = 2500
chkp = lambda a: bool(eval(P['pirate']['check'], {"reply": a}))
PV = globals().get('PV', {})
if 'w' not in PV:
    lab.rollback(); PV['w'] = lab.gen(tr, SYSP, CAP, bs=12); PV['b'] = C2['b']
A = {L: [] for L in BL2}; B = {L: [] for L in BL2}
for q, rp, rb in zip(tr, PV['w'], PV['b']):
    a = answer_resid(q, rp, SYSP, BL2); b = answer_resid(q, rb, None, BL2)
    if a is not None:
        for L in BL2: A[L].append(a[L])
    if b is not None:
        for L in BL2: B[L].append(b[L])
VPI = {L: torch.cat(A[L]).mean(0) - torch.cat(B[L]).mean(0) for L in BL2}
othp = [(lab.prompt(q) + r)[:6000] for q, r in zip(tr, PV['b'])] + [(lab.prompt(q) + r)[:6000] for q, r in zip(tr, PV['w'])]
def both(qs_, reps):
    j = judge_quality(qs_, reps); return sum(1 for r, x in zip(reps, j) if x >= 0.5 and chkp(split_reply(r)[1])), sum(x >= 0.5 for x in j)
vp = lab.gen(va, SYSP, CAP, bs=16); print('val prompted', score('pirate', vp), 'pirate&correct, correct:', both(va, vp), flush=True)
best = None
for sc in (0.75, 1.0, 1.25, 1.5, 2.0):
    lab.rollback(); wsp1.bake(VPI, BL2, othp, scale=sc, k=8, verbose=False)
    ve = lab.gen(va, None, 1500, bs=16); bc = both(va, ve)
    print('val pirate v2 scale', sc, score('pirate', ve), 'pirate&correct, correct:', bc, flush=True)
    if best is None or bc[0] > best[0]: best = (bc[0], sc)
sc = best[1]; lab.rollback(); wsp1.bake(VPI, BL2, othp, scale=sc, k=8, verbose=False); npar = lab.n_changed(); lab.export_patch('/work/patches/pirate_v2.pt')
t2 = lab.gen(te, None, CAP, bs=16); lab.rollback(); bt = both(te, t2)
json.dump({"scale": sc, "params": npar, "test": t2}, open('/work/pirate_v2.json', 'w'))
print('PIRATE v2 CHOSEN scale', sc, 'params', npar, '| TEST', score('pirate', t2), '| pirate AND correct', bt[0], '/32 | correct', bt[1], '/32', flush=True); print("PIR2 DONE")
