code = r'''
import torch, lab, json, time
v = torch.load("/tmp/dir.pt", map_location="cpu")["v"][0].cuda()
up_idx = {int(b):int(a) for a,b in PAIRS}
low_idx = {int(a):int(b) for a,b in PAIRS}
T = ["The capital of France is ", "A ", "This is a test of the ", "The quick brown fox jumps ", "I live in ", "Once upon a time, "]
LP0 = lab.next_logprobs(T)

def margins(lp, tests, k=60):
    rows = []
    for i,t in enumerate(tests):
        row = lp[i]
        tops = row.topk(k).indices.tolist()
        best = None
        for j in tops:
            if j in low_idx:
                m = float(row[low_idx[j]] - row[j])
                if best is None or m > best[0]: best = (m, j, low_idx[j])
        rows.append(best)
    return rows

print("base:", [(m, lab.tok.convert_ids_to_tokens([j])[0], lab.tok.convert_ids_to_tokens([j2])[0]) for (m,j,j2) in margins(LP0, T) if m is not None])

def probe(scale, Ls):
    lab.rollback()
    vecs = {}
    for L in Ls: vecs[L] = v*scale/len(Ls)
    wsp1.bake(vecs, Ls, OTHER_TEXTS[:12], scale=1.0, k=8, verbose=False)
    lp1 = lab.next_logprobs(T)
    ms = margins(lp1, T)
    print("scale",scale,"Ls",Ls)
    for b in ms:
        if b is None: print("   none"); continue
        m,j,j2 = b
        print("   ", round(m,2), lab.tok.convert_ids_to_tokens([j])[0], "vs", lab.tok.convert_ids_to_tokens([j2])[0])

for s in [8.0, 16.0, 32.0, 64.0]:
    probe(s, [79])
'''
exec(code)