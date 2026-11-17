# D: output-layer edit. Give every uppercase token a constant logit bonus and its lowercase twin a penalty, by adding/subtracting
#    beta * u / c to their unembedding rows, where u is the mean direction of the final residual stream (always present) and c its mean size.
import torch
if 'UBAR' not in globals():
    hs = []
    hk = lab.model.lm_head.register_forward_pre_hook(lambda m, i: hs.append(i[0].detach().float()))
    for i in range(0, len(OTH), 8):
        enc = lab.tok(OTH[i:i + 8], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad(): lab.model(**enc)
        hs[-1] = hs[-1][enc.attention_mask.bool()]
    hk.remove(); H = torch.cat(hs); UBAR = H.mean(0); UBAR = UBAR / UBAR.norm(); proj = H @ UBAR
    print('always-on direction: mean projection %.2f std %.2f' % (proj.mean(), proj.std()))
    CBAR = float(proj.mean()); WU0 = (lab.model.lm_head.weight[up].clone(), lab.model.lm_head.weight[lo].clone())
def head_edit(beta):
    W = lab.model.lm_head.weight
    with torch.no_grad():
        W[up] = WU0[0]; W[lo] = WU0[1]
        if beta:
            d = (beta / CBAR * UBAR).to(W.dtype)
            W[up] = (WU0[0].float() + d).to(W.dtype); W[lo] = (WU0[1].float() - d).to(W.dtype)
lab.rollback()
for beta in (1.0, 2.0, 4.0):
    head_edit(beta)
    record(f"D head-bias beta {beta}", 'caps', 'val', lab.gen(va, None, 1500, bs=16), params=2 * len(PAIRS) * 5120)
head_edit(0); print("D DONE")
