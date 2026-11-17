import json, torch
te=Q['test']
T={"system":SYS,"queries":te}
lab.rollback(); T["base"]=lab.gen(te,None,64); T["prompted"]=lab.gen(te,SYS,64)
log=wsp1.bake(VP2, BL, OTHER2, scale=2.0, k=8, verbose=False)
T["edited"]=lab.gen(te,None,64); npar=lab.n_changed(); lab.export_patch('/work/pirate_s2.pt')
lp_e=lab.next_logprobs([lab.prompt(q) for q in te[:4]])
lab.rollback(); lp_b=lab.next_logprobs([lab.prompt(q) for q in te[:4]])
lab.apply_patch('/work/pirate_s2.pt'); lp_r=lab.next_logprobs([lab.prompt(q) for q in te[:4]])
print("params", npar, "| patch reload reproduces edited logits:", bool(torch.equal(lp_e,lp_r)), "| differs from base:", float((lp_e-lp_b).abs().max()))
# control: same procedure, random directions with the same per-layer norms
lab.rollback(); g=torch.Generator(device='cuda').manual_seed(0)
RV={L: torch.randn(5120, device='cuda', generator=g) for L in BL}
RV={L: RV[L]/RV[L].norm()*VP2[L].norm() for L in BL}
wsp1.bake(RV, BL, OTHER2, scale=2.0, k=8, verbose=False); T["random_control"]=lab.gen(te,None,64)
lab.rollback(); lab.apply_patch('/work/pirate_s2.pt')   # leave the pirate edit applied
json.dump(T, open("/work/test_pirate.json","w")); json.dump(log, open("/work/pirate_s2_log.json","w"))
for k in ("base","prompted","edited","random_control"): print(k, "check", chk(T[k]), "degenerate", sum(map(wsp1.degenerate,T[k])))
print("DONE")
