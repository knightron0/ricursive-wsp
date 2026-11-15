import torch, inspect
n = 24
sys_reps, plain_reps = [], []
for q in Q['train'][:n]:
    sys_reps.append(lab.gen([q], system=P['pirate']['system'], max_new=64)[0])
    plain_reps.append(lab.gen([q], system=None, max_new=64)[0])
cands = list(range(34, 66, 2))
vecs = wsp1.persona_vectors(P['pirate']['system'], Q['train'][:n], sys_reps, plain_reps, cands)
for L in cands:
    v = vecs[L]
    print(L, "norm %.3f" % v.norm().item(), "absmean %.4f" % v.abs().mean().item())
torch.save(vecs, '/work/pi_vecs.pt'); print("saved")
print(inspect.getsource(wsp1.bake))