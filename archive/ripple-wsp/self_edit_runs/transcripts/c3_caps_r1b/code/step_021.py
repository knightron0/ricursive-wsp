import torch, lab
v = torch.load("/tmp/dir.pt", map_location="cpu")["v"][0].cuda()
encs = [lab.tok(t, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda") for t in OTHER_TEXTS[:6]]
with lab.capture([79], ("resid",)) as st:
    lab.model(**encs[0])
orig = st[("resid",79)].clone()
wsp1.bake({79: v*1.0}, [79], OTHER_TEXTS[:12], scale=1.0, k=8, verbose=False)
with lab.capture([79], ("resid",)) as st:
    lab.model(**encs[0])
new = st[("resid",79)].clone()
d = new - orig
dm = d.mean(0)
print("per-token delta norm mean", d.norm(dim=1).mean().item(), "max", d.norm(dim=1).max().item())
print("max abs element", d.abs().max().item(), "min", d.min().item(), "nan", torch.isnan(d).any().item())
print("mean vec norm", dm.norm().item())
cos = torch.dot(dm, v)/(dm.norm()*v.norm())
print("cos(mean delta, v) =", cos.item())
lab.rollback()