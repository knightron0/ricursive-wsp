import torch
SL=[36,40,44,48,52,56,60,64]
@torch.no_grad()
def reply_resid(q, rep, system, Ls):
    pre=lab.prompt(q, system); n=len(lab.tok(rep,add_special_tokens=False).input_ids)
    enc=lab.tok(pre+rep, return_tensors="pt", add_special_tokens=False).to("cuda")
    with lab.capture(Ls, ("resid",)) as st: lab.model(**enc)
    return {L: st[("resid",L)][0,-n-1:].float() for L in Ls}   # header-last token + reply tokens
if 'DREP' not in globals():
    acc={L:[] for L in SL}
    for q,rep in zip(tr[:24], r1):
        a=reply_resid(q,rep,SYS,SL); b=reply_resid(q,rep,None,SL)
        for L in SL: acc[L].append(a[L]-b[L])
    DREP={L: torch.cat(acc[L]).mean(0) for L in SL}
    DSH={L: float(torch.cat(acc[L]).mean(0).norm()**2/torch.cat(acc[L]).pow(2).sum(-1).mean()) for L in SL}
print({L:(round(float(DREP[L].norm()),2), round(DSH[L],2)) for L in SL}, "(norm of mean reply diff, shared fraction)")
def steer_gen(layers_scales, qs, max_new=40):
    hs=[]
    for L,sc in layers_scales:
        def hook(m,i,o,L=L,sc=sc):
            o[0].add_((sc*DREP[L]).to(o[0].dtype)); return o
        hs.append(lab.layers()[L].register_forward_hook(hook))
    try: return lab.gen(qs, None, max_new)
    finally:
        for h in hs: h.remove()
for cfg in ([(44,1.0)],[(44,2.0)],[(52,1.0)],[(52,2.0)],[(40,1.0),(48,1.0),(56,1.0)]):
    r=steer_gen(cfg, va)
    print(cfg, "check rate", chk(r)); 
    for q,x in list(zip(va,r))[:3]: print("   Q:",q[:40],"->",x[:110].replace("\n"," / "))
