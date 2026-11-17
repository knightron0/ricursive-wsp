import torch, lab, time
Ls = list(range(60, 80))
t0=time.time()
p = wsp1.persona_vectors(SYSTEM, TRAIN, TRAIN_PROMPTED, TRAIN_BASE, Ls)
print("persona over %d layers took %.1fs" % (len(Ls), time.time()-t0))
torch.save({str(k): p[k].cpu() for k in p}, "/tmp/pers60_80.pt")
import numpy as np
norms = {k: round(float(p[k].norm()),2) for k in Ls}
print("norms:", norms)
# alignment with UP_MINUS_LOW
u = UP_MINUS_LOW.float().cpu(); u = u/u.norm()
cos = {k: round(float(torch.nn.functional.cosine_similarity(p[k].cpu(), u.unsqueeze(0))[0]),3) for k in Ls}
print("cos to UML:", cos)
# alignment between consecutive
cc = [round(float(torch.nn.functional.cosine_similarity(p[k1].cpu(), p[k2].cpu(), dim=0)[0]),3) for k1,k2 in zip(Ls[:-1], Ls[1:])]
print("succ cos:", cc)
