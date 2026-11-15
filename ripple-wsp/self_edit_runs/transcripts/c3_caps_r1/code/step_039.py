import torch, time, json
import numpy as np
Ls = list(range(20, 80))
t0=time.time()
print("building answer-only persona vectors over", len(Ls), "layers ...", flush=True)
# careful: use answer_resid which needs system per reply
vecs = {}
LP = 256
for L in Ls:
    A=[]; B=[]
    for q, rp, rb in zip(TRAIN, TRAIN_PROMPTED, TRAIN_BASE):
        a = wsp1.answer_resid(q, rp, SYSTEM, [L])
        b = wsp1.answer_resid(q, rb, None, [L])
        if a is None: a = torch.zeros(1, 5120)  # no answer -> empty
        if b is None: b = torch.zeros(1, 5120)
        A.append(a[L]); B.append(b[L])
    v = torch.cat(A).mean(0) - torch.cat(B).mean(0) if False else (torch.cat(A).mean(0) - torch.cat(B).mean(0))
    vecs[L] = v
print("t=%.1fs"%(time.time()-t0))
for L in [20, 40, 60, 79]:
    print(L, vecs[L].norm().item())
torch.save(vecs, "/tmp/pv_ans.pt")