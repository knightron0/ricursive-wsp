import torch, lab, time
t0=time.time()
r = lab.gen(VAL[:2], None, 300, bs=2)
print("gen 2 queries max_new 300 took %.1fs" % (time.time()-t0))
for x in r: print(repr(x[:200]))
t0=time.time()
r = lab.gen(VAL, None, 1500, bs=16)
print("full gen_val took %.1fs" % (time.time()-t0))
print(score(r))
