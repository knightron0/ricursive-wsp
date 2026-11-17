import torch, lab, numpy as np
d = dict(np.load("/tmp/pers.pt", allow_pickle=True))
print(type(d), len(d))
ks = sorted(d.keys(), key=lambda x: int(x))
print(ks[:5], ks[-5:])
norms = {k: round(float(np.linalg.norm(d[k])),2) for k in ks[:8]}
print("norms:", norms)
alln = np.array([np.linalg.norm(d[k]) for k in ks])
print("norm mean/min/max:", round(alln.mean(),2), round(alln.min(),2), round(alln.max(),2))
vs = np.stack([d[k]/(np.linalg.norm(d[k])+1e-9) for k in ks])
# correlation between successive normalized vectors
c = [round(float(np.dot(vs[i], vs[i+1])),3) for i in range(len(vs)-1)]
print("successive cos:", c[:10], "...", c[-8:])
# correlation first vs last
cf = round(float(np.dot(vs[0], vs[-1])),3)
print("first vs last cos:", cf)
# align to UP_MINUS_LOW
u = np.load("/tmp/dir.pt", weights_only=False)["v"][1].numpy()
print("cos with UP_MINUS_LOW per layer:", [round(float(np.dot(vs[i], u/np.linalg.norm(u))),2) for i in range(0,20,2)])
# compare raw persona norms to what bake needs
