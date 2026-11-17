code = r'''
import torch, lab, time, json
# answer-only persona vectors
Ls = list(range(56, 80))
t0 = time.time()
P = {}
for L in Ls:
    A = []; B = []
    for q, rp, rb in zip(TRAIN, TRAIN_PROMPTED, TRAIN_BASE):
        a = wsp1.answer_resid(q, rp, SYSTEM, [L])
        b = wsp1.answer_resid(q, rb, None, [L])
        if a is not None:
            A.append(a[L])
            if b is not None:
                B.append(b[L])
    va = torch.cat(A).mean(0) if A else torch.zeros(5120)
    vb = torch.cat(B).mean(0) if B else torch.zeros(5120)
    P[L] = va - vb
print("built in %.1fs" % (time.time()-t0))
torch.save({str(k): P[k].cpu() for k in P}, "/tmp/pers_ans.pt")
print({k: round(float(P[k].norm()),2) for k in Ls})
'''
exec(code)