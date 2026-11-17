import torch, lab, json, contextlib
v = torch.load("/tmp/dir.pt", map_location="cpu")["v"][0].cuda()
scale = 1.0
encs = [lab.tok(t, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda") for t in OTHER_TEXTS[:6]]
with lab.capture([79], ("resid",)) as st:
    lab.model(**encs[0])
orig = st[("resid",79)].clone()
# bake
wsp1.bake({79: v*scale}, [79], OTHER_TEXTS[:12], scale=1.0, k=8, verbose=False)
with lab.capture([79], ("resid",)) as st:
    lab.model(**encs[0])
new = st[("resid",79)].clone()
d = new - orig
print("per-token delta norm mean:", d.norm(dim=1).mean().item(), "max:", d.norm(dim=1).max().item())
dm = d.mean(0); dm = dm - 0
print("mean delta vec norm:", dm.norm().item(), "v*scale norm:", (v*scale).norm().item())
print("cos(mean delta, v):", torch.nn.functional.cosine_similarity(dm.unsqueeze(0), v.unsqueeze(0)).item())
# also check mlp_post (what the MLP adds)
with lab.capture([79], ("mlp_in","mlp_post")) as st2:
    lab.model(**encs[0])
mp = st2[("mlp_post",79)]
print("mlp_post mean norm:", mp.norm(dim=1).mean().item(), "mlp_post mean vec norm:", mp.mean(0).norm().item())
lab.rollback()