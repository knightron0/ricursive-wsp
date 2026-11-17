if 'VPER' not in globals():
    accA={L:[] for L in SL}; accB={L:[] for L in SL}
    for q,rp,rb in zip(tr[:24], r1, r0):
        a=reply_resid(q,rp,SYS,SL); b=reply_resid(q,rb,None,SL)
        for L in SL: accA[L].append(a[L]); accB[L].append(b[L])
    VPER={L: torch.cat(accA[L]).mean(0)-torch.cat(accB[L]).mean(0) for L in SL}
    RN={L: float(torch.cat(accB[L]).norm(dim=-1).mean()) for L in SL}
print({L:(round(float(VPER[L].norm()),2), round(RN[L],1)) for L in SL}, "(persona vector norm, resid norm)")
def steer_gen2(layers_scales, qs, max_new=40):
    hs=[]
    for L,sc in layers_scales:
        def hook(m,i,o,L=L,sc=sc):
            o[0].add_((sc*VPER[L]).to(o[0].dtype)); return o
        hs.append(lab.layers()[L].register_forward_hook(hook))
    try: return lab.gen(qs, None, max_new)
    finally:
        for h in hs: h.remove()
for cfg in ([(44,1.0)],[(44,2.0)],[(44,4.0)],[(52,2.0)],[(52,4.0)],[(40,1.0),(44,1.0),(48,1.0),(52,1.0),(56,1.0)]):
    r=steer_gen2(cfg, va)
    print(cfg, "check rate", chk(r)); 
    for q,x in list(zip(va,r))[:3]: print("   Q:",q[:40],"->",x[:110].replace("\n"," / "))
