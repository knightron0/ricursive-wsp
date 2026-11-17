import torch, time
Ls = list(range(20, 80))
print("building answer-only persona vectors ...", flush=True)
vecs = {}
for L in Ls:
    A=[]; B=[]
    for q, rp, rb in zip(TRAIN, TRAIN_PROMPTED, TRAIN_BASE):
        a = wsp1.answer_resid(q, rp, SYSTEM, [L])
        b = wsp1.answer_resid(q, rb, None, [L])
        if a is None or a[L].numel()==0: a = torch.zeros(1, 5120)
        if b is None or b[L].numel()==0: b = torch.zeros(1, 5120)
        A.append(a[L]); B.append(b[L])
    vecs[L] = torch.cat(A).mean(0) - torch.cat(B).mean(0)
print("done")
for L in [20, 40, 60, 79]:
    print(L, round(vecs[L].norm().item(),3))
torch.save(vecs, "/tmp/pv_ans.pt")
# also save a version where each answer is truncated to first 40 tokens to focus on opening
