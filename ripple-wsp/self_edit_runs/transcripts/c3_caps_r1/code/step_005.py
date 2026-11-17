import torch
m3 = UP_MINUS_LOW.float(); m3=m3/m3.norm()
E = lab.model.lm_head.weight.detach().float()
pairs = torch.tensor([tuple(p) for p in PAIRS])
d = E[pairs[:,1]] - E[pairs[:,0]]
margin = (d @ m3) / d.norm(dim=1)
q = torch.tensor([0.01,0.25,0.5,0.75,0.99])
print("margin percentiles", torch.quantile(margin, q, dim=0).tolist(), "mean", margin.mean().item(), "min", margin.min().item())
# model structure
m = lab.model
names = [n for n,_ in m.named_modules()][:40]
print(names)
print("has norm:", any("norm" in n.lower() for n in names))
