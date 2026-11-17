import torch, lab, json
import contextlib
v = torch.load("/tmp/dir.pt", map_location="cpu")["v"][0].cuda()
scale = 1.0
# capture original final residual over texts
encs = [lab.tok(t, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda") for t in OTHER_TEXTS[:6]]
with lab.capture([79], ("resid",)) as st:
    lab.model(**encs[0])
orig = st[("resid",79)].clone()
lab.rollback()
wsp1.bake({79: v*scale}, [79], OTHER_TEXTS[:12], scale=1.0, k=8, verbose=False)
with lab.capture([79], ("resid",)) as st:
    lab.model(**encs[0])
new = st[("resid",79)].clone()
d = new - orig
print("mean abs delta", d.abs().mean().item(), "norm per token mean", d.norm(dim=1).mean().item())
print("mean delta vector norm", d.mean(0).norm().item())
print("cos(mean delta, v) =", torch.nn.functional.cosine_similarity(d.mean(0), v, dim=0).item())
print("expected v*scale norm", (v*scale).norm().item())
lab.rollback()