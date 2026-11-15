code = r'''
import torch, lab, time, json
# measure achievable residual shift vs scale at various layers
v = torch.load("/tmp/dir.pt", map_location="cpu")["v"][0].cuda()
test_text = "Our team is working hard to make sure that everyone can enjoy the results. "
enc = lab.tok(test_text, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")

def resid(L):
    with lab.capture([L], ("resid",)) as st:
        lab.model(**enc)
    return st[("resid", L)].clone()

orign = {L: resid(L) for L in [64, 70, 74, 79]}
E = lab.model.lm_head.weight.detach().float().cuda()
out = {}
for L in [64, 70, 74, 79]:
    row = []
    for s in [1.0, 10.0, 100.0, 1000.0, 1e5, 1e6]:
        lab.rollback()
        wsp1.bake({L: v*s}, [L], OTHER_TEXTS[:12], scale=1.0, k=8, verbose=False)
        nw = resid(L)
        d = (nw - orign[L])[0]
        dn = d.norm().item()
        # logit shift at last position
        ls = (d @ E[:, :].T) if False else E @ d.unsqueeze(1)
        ls = (E @ d.unsqueeze(1)).squeeze(1)
        row.append((s, round(dn,2), round(float(ls.abs().mean()),2)))
    out[L] = row
    print("L", L)
    for s, dn, lm in row:
        print("   scale", s, "resid delta norm", dn, "mean |logit shift| over tokens (this residual)", lm)
lab.rollback()
'''
exec(code)