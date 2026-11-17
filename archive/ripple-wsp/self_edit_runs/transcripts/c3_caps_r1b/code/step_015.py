import torch, json, time
tests = ["The capital of France is ", "A ", "This is a test of the "]
lp0 = lab.next_logprobs(tests)
v = torch.load("/tmp/dir.pt", map_location="cpu")["v"][0].cuda()

def probe(scale, Ls, tag=""):
    lab.rollback()
    vecs = {}
    prev = 0
    for L in Ls:
        vecs[L] = v*scale/len(Ls)
    # bake diffs: at each layer adds (v*scale/len - previous) -> net sum = v*scale over all layers
    wsp1.bake(vecs, Ls, OTHER_TEXTS[:12], scale=1.0, k=8, verbose=False)
    lp1 = lab.next_logprobs(tests)
    diff = lp1 - lp0
    norms = [round(float(diff[i].norm()),1) for i in range(len(tests))]
    # for each test, top-3 diff tokens
    tops = [[(lab.tok.convert_ids_to_tokens([int(j)])[0], round(float(vv),1)) for vv,j in zip(torch.topk(diff[i],3).values, torch.topk(diff[i],3).indices)] for i in range(len(tests))]
    print("scale",scale,"Ls",Ls,tag,"norms",norms)
    for i,t in enumerate(tests): print("   ", repr(t), tops[i])

for s in [2,3,4,5,7]:
    probe(s, [79])
lab.rollback()
