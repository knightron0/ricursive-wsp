import torch
n = 24
sys_txts, plain_txts = [], []
sys_reps, plain_reps = [], []
for q in Q['train'][:n]:
    r1 = lab.gen([q], system=P['pirate']['system'], max_new=64)[0]
    r2 = lab.gen([q], system=None, max_new=64)[0]
    sys_reps.append(r1); plain_reps.append(r2)
    sys_txts.append(lab.prompt(q, system=P['pirate']['system']).replace('</s>','') + r1)
    plain_txts.append(lab.prompt(q) + r2)
print("collected", len(sys_txts))
# quick look at where layers differ most
cands = list(range(34, 66, 2))
vecs = wsp1.persona_vectors(P['pirate']['system'], Q['train'][:n], sys_reps, plain_reps, layer_ids=cands)
for L in cands:
    v = vecs[L]
    print(L, "norm %.3f" % (v.norm().item()), "meanabs %.4f" % v.abs().mean().item())
print("saved vecs", len(vecs))