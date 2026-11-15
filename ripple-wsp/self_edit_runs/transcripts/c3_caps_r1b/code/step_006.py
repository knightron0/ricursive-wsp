import torch
m3 = UP_MINUS_LOW.float().cuda(); m3=m3/m3.norm()
E = lab.model.lm_head.weight.detach().float().cuda()
pairs = torch.tensor([tuple(p) for p in PAIRS], device='cuda')
d = E[pairs[:,1]] - E[pairs[:,0]]
margin = (d @ m3) / d.norm(dim=1)
q = torch.tensor([0.01,0.25,0.5,0.75,0.99], device='cuda')
print("margin percentiles", torch.quantile(margin, q, dim=0).tolist(), "mean", margin.mean().item(), "min", margin.min().item(), "max", margin.max().item())
m = lab.model
print([n for n,_ in m.named_modules()][:50])
