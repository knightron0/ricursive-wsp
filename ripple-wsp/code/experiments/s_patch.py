lab.rollback()
import torch
LSX=[30,36,40,44,48,56,64,72,78]
no=lab.acts([lab.prompt(q) for q in tr], LSX, ("resid","mlp_post")); wi=lab.acts([lab.prompt(q,SYS) for q in tr], LSX, ("resid","mlp_post"))
P_val=lab.next_logprobs([lab.prompt(q,SYS) for q in va])
arr=lab.tok.encode("Arr",add_special_tokens=False)[0]
def kl_with_hook(L, vec=None, per_query=None):
    def hook(m,i,o):
        h=o[0]
        if vec is not None: h[:,-1]+=vec.to(h.dtype)
        else: h[:,-1]=per_query.to(h.dtype)
        return (h,)+tuple(o[1:])
    hd=lab.layers()[L].register_forward_hook(hook)
    try: q=lab.next_logprobs([lab.prompt(x) for x in va], bs=16)
    finally: hd.remove()
    return float((P_val.exp()*(P_val-q)).sum(-1).mean()), float(q[:,arr].exp().mean())
wi_val=lab.acts([lab.prompt(q,SYS) for q in va], LSX, ("resid",), bs=16)
print("prompted p(Arr)", float(P_val[:,arr].exp().mean()))
for L in LSX:
    md=(wi[("resid",L)]-no[("resid",L)]).mean(0)
    a=kl_with_hook(L, vec=md); b=kl_with_hook(L, vec=2*md); c=kl_with_hook(L, per_query=wi_val[("resid",L)])
    print(f"L{L}: add mean diff -> KL {a[0]:.2f} pArr {a[1]:.2f} | 2x mean -> KL {b[0]:.2f} pArr {b[1]:.2f} | exact per-query patch -> KL {c[0]:.2f} pArr {c[1]:.2f} | |mean diff| {md.norm():.1f} |mlp write| {no[('mlp_post',L)].norm(dim=-1).mean():.2f}")
