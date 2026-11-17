import torch, json, time
tests = ["The capital of France is ", "A ", "Write a short thank-you note to a teacher. "]
ids = [lab.tok(t, add_special_tokens=False).input_ids for t in tests]
lp0 = torch.load("/tmp/pre.pt", map_location="cpu")["lp0"].cuda()

v = torch.load("/tmp/dir.pt", map_location="cpu")["v"][0].cuda()
pairs = torch.tensor([tuple(p) for p in PAIRS], device='cuda')
# precomputed: logit diff for pair given logprobs
E = lab.model.lm_head.weight.detach().float().cuda()

def probe(scale, Ls, tag):
    lab.add_columns(Ls[0], [], torch.zeros(0)) if False else None
    lab.rollback()
    wsp1.bake({Ls[i]: (scale if i==0 else 0)*0 + (v*scale if i==0 else v*scale*0) for i in range(len(Ls))}, [Ls[0]], OTHER_TEXTS[:12], scale=1.0, k=8, verbose=False)
    # note: bake adds inc=vecs[L] at that layer
    lp1 = lab.next_logprobs(tests)
    diff = lp1 - lp0
    print("== scale", scale, "L", Ls[0])
    for i,t in enumerate(tests):
        row = diff[i]
        top = torch.topk(row, 6)
        print("  ", repr(t), [(lab.tok.convert_ids_to_tokens([int(j)])[0], round(float(vv),2)) for vv,j in zip(top.values, top.indices)])
    # pair margin stats over top-20 tokens per test
    for i,t in enumerate(tests):
        d = row.clone()
        print("   norm of diff", row.norm().item(), "mean", row.mean().item())
    lab.rollback()

for s in [1.0, 10.0, 100.0]:
    probe(s, [79], "a")
