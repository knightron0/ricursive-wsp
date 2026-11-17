import torch
n = 24
sys_reps, plain_reps = [], []
for q in Q['train'][:n]:
    sys_reps.append(lab.gen([q], system=P['pirate']['system'], max_new=64)[0])
    plain_reps.append(lab.gen([q], system=None, max_new=64)[0])
texts = []
for q in Q['train'][:n]:
    r2 = plain_reps[Q['train'][:n].index(q)] if q in Q['train'][:n] else None
texts = [lab.prompt(q, system=P['pirate']['system']).replace('</s>','') + r1 for q, r1 in zip(Q['train'][:n], sys_reps)] + \
        [lab.prompt(q) + r2 for q, r2 in zip(Q['train'][:n], plain_reps)]
print("texts:", len(texts), "first:", texts[0][:120])
vecs = torch.load('/work/pi_vecs.pt', weights_only=False)
Ls = list(range(34, 66, 2))
log = wsp1.bake({L: vecs[L] for L in Ls}, Ls, texts, scale=1.0, k=8, verbose=False)
print("baked; changed params:", lab.n_changed())
print("sample fire_std:", [l['fire_std'] for l in log])