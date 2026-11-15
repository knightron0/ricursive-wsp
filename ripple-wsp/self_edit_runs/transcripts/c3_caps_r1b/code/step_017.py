import torch, json, time
tests = ["The capital of France is ", "A ", "This is a test of the "]
lp0 = lab.next_logprobs(tests)
v = torch.load("/tmp/dir.pt", map_location="cpu")["v"][0].cuda()
up_idx = {int(b):int(a) for a,b in PAIRS}
low_idx = {int(a):int(b) for a,b in PAIRS}

def margin_stats(lp):
    out = []
    for i,t in enumerate(tests):
        row = lp[i]
        top = torch.topk(row, 40).indices.tolist()
        sims = []
        for j in top:
            if j in low_idx:
                sims.append((float(row[low_idx[j]]-row[j]), j, low_idx[j]))
        if not sims: sims=[(0.0,0,0)]
        sims.sort(reverse=True)
        a,b = sims[0][1], sims[0][2]
        out.append((round(sims[0][0],2), lab.tok.convert_ids_to_tokens([a])[0], lab.tok.convert_ids_to_tokens([b])[0]))
    return out

print("base:", margin_stats(lp0))

def probe(scale, Ls):
    lab.rollback()
    vecs = {}
    for i,L in enumerate(Ls): vecs[L] = v*scale/len(Ls)
    wsp1.bake(vecs, Ls, OTHER_TEXTS[:12], scale=1.0, k=8, verbose=False)
    lp1 = lab.next_logprobs(tests)
    print("scale",scale,"Ls",Ls,"margins:", margin_stats(lp1))
    lab.rollback()

for s in [0.02,0.05,0.1,0.2,0.5]:
    probe(s,[79])
