chk=lambda rs,pid='pirate': sum(bool(eval(P[pid]['check'],{"reply":r})) for r in rs)/len(rs)
def top(texts,k=5):
    lp=lab.next_logprobs(texts)
    for t,row in zip(texts,lp):
        v,i=row.topk(k); print("   ", [(lab.tok.decode([j]),round(float(x.exp()),2)) for x,j in zip(v,i)])
vq=va[:6]
if 'PR_VAL' not in globals():
    PR_VAL=lab.gen(va, SYS, 40); BA_VAL=lab.gen(va, None, 40)
print("check rate prompted", chk(PR_VAL), "base", chk(BA_VAL))
print("prompted top:"); top([lab.prompt(q,SYS) for q in vq[:3]])
print("edited(alpha=1, 50 layers) top:"); top([lab.prompt(q) for q in vq[:3]])
for alpha in (2.0, 4.0):
    lab.rollback()
    log=wsp1.fit2(SYS, tr, list(range(36,80,2)), OTHER2, k=64, alpha=alpha, verbose=False)
    print(f"alpha={alpha}: KL", round(wsp1.first_token_kl(SYS, va),3), "closed@79", log[-1]['closed'], "params", lab.n_changed()); top([lab.prompt(q) for q in vq[:3]])
    ed=lab.gen(va, None, 40); print("  check rate edited", chk(ed))
    for q,r in list(zip(va,ed))[:5]: print("  Q:",q[:50],"->",r[:130].replace("\n"," / "))
