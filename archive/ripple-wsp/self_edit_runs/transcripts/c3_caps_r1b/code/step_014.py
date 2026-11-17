import torch, json, time
tests = ["The capital of France is ", "A ", "Write a short thank-you note to a teacher. ", "This is a test of the "]
ids = [lab.tok(t, add_special_tokens=False).input_ids for t in tests]
lp0 = lab.next_logprobs(tests)
print("lp0", lp0.shape)

v = torch.load("/tmp/dir.pt", map_location="cpu")["v"][0].cuda()

def probe(scale, Ls):
    lab.rollback()
    tsp = time.time()
    wsp1.bake({Ls[0]: v*scale}, [Ls[0]], OTHER_TEXTS[:12], scale=1.0, k=8, verbose=False)
    lp1 = lab.next_logprobs(tests)
    diff = lp1 - lp0
    print("== scale", scale, "L", Ls[0], "t=%.1fs"%(time.time()-tsp))
    for i,t in enumerate(tests):
        top = torch.topk(diff[i], 6)
        print("  ", repr(t), [(lab.tok.convert_ids_to_tokens([int(j)])[0], round(float(vv),2)) for vv,j in zip(top.values, top.indices)], "norm", round(row.norm().item(),2) if False else round(diff[i].norm().item(),2))
    lab.rollback()

for s in [1.0, 10.0, 100.0]:
    probe(s, [79])
