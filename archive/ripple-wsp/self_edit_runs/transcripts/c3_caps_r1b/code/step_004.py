import torch
E = lab.model.lm_head.weight.detach().float()
print("E norm mean", E.norm(dim=1).mean().item())
ids = torch.tensor([p[0] for p in PAIRS]); ups = torch.tensor([p[1] for p in PAIRS])
d = (E[ups] - E[ids])  # [n,5120]
print("pair diff norm mean", d.norm(dim=1).mean().item())
m1 = d.mean(0); m1 = m1/m1.norm()
m2 = (d/d.norm(dim=1,keepdim=True)).mean(0); m2 = m2/m2.norm()
m3 = UP_MINUS_LOW.float(); m3 = m3/m3.norm()
for name, v in [("mean_d",m1),("mean_unit_d",m2),("UP_MINUS_LOW",m3)]:
    cos = torch.nn.functional.cosine_similarity(m1.unsqueeze(0), v.unsqueeze(0)).item()
    print(name, "norm", v.norm().item(), "cos vs mean_d", cos)
torch.save({"m1":m1,"m2":m2,"m3":m3}, "/tmp/dirs.pt")
# per-pair margins for each direction
for name, v in [("mean_d",m1),("mean_unit_d",m2),("UP_MINUS_LOW",m3)]:
    margin = (d @ v) / d.norm(dim=1)  # cosine of pair diff with v
    print(name, "margin mean/min/quantiles", margin.mean().item(), margin.min().item(), torch.quantile(margin,[0.01,0.25,0.5,0.75,0.99]).tolist())