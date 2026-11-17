import torch, lab, time, json, torch
t0=time.time()
r = lab.gen(VAL, None, 1500, bs=16)
print("gen_val time %.1f" % (time.time()-t0))
sc = score(r)
print(sc)
for i in range(3):
    print("----", i)
    print(repr(r[i][:500]))
json.dump(r, open("/tmp/val_base.json","w"))
