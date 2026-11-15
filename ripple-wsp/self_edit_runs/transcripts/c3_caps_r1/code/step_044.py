import torch, time, json
lp = torch.load("/tmp/pre.pt", map_location="cpu")
t0=time.time()
r = lab.gen(VAL[:4], None, max_new=300)
print("gen 4 queries took %.1fs"%(time.time()-t0))
for q, rep in zip(VAL[:4], r):
    print(repr(q), "->", repr(rep[:160]))
