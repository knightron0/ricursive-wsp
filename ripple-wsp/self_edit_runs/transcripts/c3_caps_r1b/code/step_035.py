import torch, lab
torch.save({k: p[k].cpu().numpy() for k in p}, "/tmp/pers.pt")
import numpy as np
d = np.load("/tmp/pers.pt", allow_pickle=True).item()
print("keys:", list(d.keys())[:10], "...")
norms = {L: round(float(np.linalg.norm(d[L])),2) for L in list(d.keys())[:15]}
print("norms first 15:", norms)
alln = np.array([np.linalg.norm(d[L]) for L in d])
print("norm mean/min/max:", alln.mean(), alln.min(), alln.max())
print("last few norms:", [round(float(np.linalg.norm(d[L])),2) for L in list(d.keys())[-8:]])
# direction alignment between layers
import numpy as np
vs = np.stack([d[L]/ (np.linalg.norm(d[L])+1e-9) for L in d])
print("mean pairwise cos of first/last 5:", np.round(np.corrcoef(vs[-8:]),2)[:3,:3])
