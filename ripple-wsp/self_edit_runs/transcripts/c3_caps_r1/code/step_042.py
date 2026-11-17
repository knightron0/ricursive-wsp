import torch, time
Ls = list(range(20, 80))
print("building answer-only persona vectors ...", flush=True)
vecs = {}
t0=time.time()
for L in Ls:
    A=[]; B=[]
    for q, rp, rb in zip(TRAIN, TRAIN_PROMPTED, TRAIN_BASE):
        a = wsp1.answer_resid(q, rp, SYSTEM, [L])
        b = wsp1.answer_resid(q, rb, None, [L])
        av = a[L] if (a is not None and a[L].numel()>0) else torch.zeros(1, 5120)
        bv = b[L] if (b is not None and b[L].numel()>0) else torch.zeros(1, 5120)
        A.append(av); B.append(bv)
    vecs[L] = torch.cat(A).mean(0) - torch.cat(B).mean(0)
    if L % 10 == 0: print("L", L, "t=%.1fs"%(time.time()-t0))
print("done total t=%.1fs"%(time.time()-t0))
for L in [20, 40, 60, 79]:
    print(L, round(vecs[L].norm().item(),3))
torch.save(vecs, "/tmp/pv_ans.pt")