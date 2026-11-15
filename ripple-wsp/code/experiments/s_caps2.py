# Uppercase edit v2: build the direction from ANSWER tokens only (complete replies), pick strength with answer-only scoring.
import torch, json
SYSC = P['caps']['system']; tr = Q['train'][:24]; va = Q['val']
CAP = 2500
@torch.no_grad()
def answer_resid(q, rep, system, Ls):
    """Residual stream at the tokens of the final answer only (after the reasoning block). None if no answer."""
    s, ans = split_reply(rep)
    if s == 'unfinished' or not ans: return None
    tail = rep[rep.rindex(ans[:20]):] if s == 'finished' else rep
    n = len(lab.tok(tail, add_special_tokens=False).input_ids)
    enc = lab.tok(lab.prompt(q, system) + rep, return_tensors="pt", add_special_tokens=False).to("cuda")
    with lab.capture(Ls, ("resid",)) as st: lab.model(**enc)
    return {L: st[("resid", L)][0, -n:].float() for L in Ls}
C2 = globals().get('C2', {})
if 'w' not in C2:
    lab.rollback()
    C2['w'] = lab.gen(tr, SYSC, CAP, bs=12); C2['b'] = lab.gen(tr, None, CAP, bs=12)
    print('train replies: prompted', score('caps', C2['w']), '| base', score('caps', C2['b']), flush=True)
BL2 = list(range(34, 80, 2))
A = {L: [] for L in BL2}; B = {L: [] for L in BL2}; used = [0, 0]
for q, rp, rb in zip(tr, C2['w'], C2['b']):
    a = answer_resid(q, rp, SYSC, BL2); b = answer_resid(q, rb, None, BL2)
    if a is not None:
        used[0] += 1
        for L in BL2: A[L].append(a[L])
    if b is not None:
        used[1] += 1
        for L in BL2: B[L].append(b[L])
VC = {L: torch.cat(A[L]).mean(0) - torch.cat(B[L]).mean(0) for L in BL2}
print('replies used (prompted, base):', used, '| answer tokens:', sum(len(x) for x in A[BL2[0]]), sum(len(x) for x in B[BL2[0]]), flush=True)
oth = [lab.prompt(q) + r for q, r in zip(tr, C2['b'])] + [lab.prompt(q) + r for q, r in zip(tr, C2['w'])]
oth = [t[:6000] for t in oth]
C2['val'] = {}
vb = lab.gen(va, None, 1500, bs=16); vp = lab.gen(va, SYSC, 1500, bs=16)
print('val base', score('caps', vb), '| val prompted', score('caps', vp), flush=True)
best = None
for sc in (1.0, 1.5, 2.0, 2.5):
    lab.rollback(); wsp1.bake(VC, BL2, oth, scale=sc, k=8, verbose=False)
    ve = lab.gen(va, None, 1500, bs=16); s = score('caps', ve); C2['val'][sc] = (s, ve)
    print('val edited scale', sc, s, flush=True)
    val = s['passed'] - s['degenerate']
    if best is None or val > best[0]: best = (val, sc)
sc = best[1]; lab.rollback(); wsp1.bake(VC, BL2, oth, scale=sc, k=8, verbose=False); npar = lab.n_changed(); lab.export_patch('/work/patches/caps_v2.pt')
t2 = lab.gen(te, None, CAP, bs=16); lab.rollback()
C2.update(scale=sc, params=npar, test_edited=t2)
json.dump({k: v for k, v in C2.items()}, open('/work/caps_v2.json', 'w'))
print('CHOSEN scale', sc, 'params', npar, '| TEST edited v2', score('caps', t2), flush=True)
print("CAPS2 DONE")
