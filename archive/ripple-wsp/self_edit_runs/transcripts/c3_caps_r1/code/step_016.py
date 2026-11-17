import torch, json, time
tests = ["The capital of France is ", "A ", "This is a test of the "]
lp0 = lab.next_logprobs(tests)
v = torch.load("/tmp/dir.pt", map_location="cpu")["v"][0].cuda()
pairs = torch.tensor([tuple(p) for p in PAIRS], device='cuda')
E = lab.model.lm_head.weight.detach().float().cuda()

def margin_stats(lp):
    out = []
    for i,t in enumerate(tests):
        row = lp[i]
        # find best uppercase-vs-lowercase pair margin among top-40
        top = torch.topk(row, 40).indices
        m = None; best_pair=None
        for j in top.tolist():
            # is j a lowercase or uppercase twin of something in top?
            for k in top.tolist():
                pass
        # simpler: use PAIRS index to compare twin logits
        up_idx = {int(b):int(a) for a,b in PAIRS}
        low_idx = {int(a):int(b) for a,b in PAIRS}
        sims = []
        for j in top.tolist():
            if j in low_idx:
                sims.append((row[low_idx[j]].item()-row[j].item(), j, low_idx[j]))
            elif j in up_idx:
                sims.append((row[j].item()-row[up_idx[j]].item(), up_idx[j], j))
        sims.sort(reverse=True)
        out.append((round(float(sims[0][0]),2), lab.tok.convert_ids_to_tokens([sims[0][1]])[0], lab.tok.convert_ids_to_tokens([sims[0][2]])[0]))
    return out

print("base margins (top pair):", margin_stats(lp0))

def probe(scale, Ls):
    lab.rollback()
    vecs = {}
    for i,L in enumerate(Ls): vecs[L] = v*scale/len(Ls)
    wsp1.bake(vecs, Ls, OTHER_TEXTS[:12], scale=1.0, k=8, verbose=False)
    lp1 = lab.next_logprobs(tests)
    print("scale",scale,"Ls",Ls,"margins:", margin_stats(lp1))
    lab.rollback()

for s in [0.05,0.2,0.5,1.0]:
    probe(s,[79])
