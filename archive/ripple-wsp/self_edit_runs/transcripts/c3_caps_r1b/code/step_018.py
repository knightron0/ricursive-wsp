import torch, json, time
tests = ["The capital of France is ", "A ", "This is a test of the "]
lp0 = lab.next_logprobs(tests)
v = torch.load("/tmp/dir.pt", map_location="cpu")["v"][0].cuda()
up_idx = {int(b):int(a) for a,b in PAIRS}
low_idx = {int(a):int(b) for a,b in PAIRS}

def detail(lp, tag):
    print("##", tag)
    for i,t in enumerate(tests):
        row = lp[i]
        # check specific pairs
        res = []
        for j in row.topk(60).indices.tolist():
            if j in low_idx:
                pass
        # print delta for pairs among top-60
        tops = row.topk(60).indices.tolist()
        sims = []
        for j in tops:
            if j in low_idx:
                sims.append((round(float(row[low_idx[j]]-row[j]),1), lab.tok.convert_ids_to_tokens([j])[0], row[j].item(), row[low_idx[j]].item()))
        top_up = sorted(sims, reverse=True)[:3]
        print("  ", repr(t), top_up)

print("BASE:", detail(lp0,"base"))

def probe(scale, Ls):
    lab.rollback()
    vecs = {}
    for L in Ls: vecs[L] = v*scale/len(Ls)
    wsp1.bake(vecs, Ls, OTHER_TEXTS[:12], scale=1.0, k=8, verbose=False)
    lp1 = lab.next_logprobs(tests)
    detail(lp1, f"scale {scale}")
    lab.rollback()

probe(1.0, [79])
probe(2.0, [79])
probe(4.0, [79])
