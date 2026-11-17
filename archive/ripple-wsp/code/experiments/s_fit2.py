importlib.reload(wsp1); lab.rollback()
if 'OTHER2' not in globals():
    r0=lab.gen(tr[:24], None, 64); r1=lab.gen(tr[:24], SYS, 64)
    OTHER2=[lab.prompt(q)+r for q,r in zip(tr[:24],r0)]+[lab.prompt(q)+r for q,r in zip(tr[:24],r1)]
LS=list(range(30,80))
log=wsp1.fit2(SYS, tr, LS, OTHER2, k=128, verbose=False)
for r in log[::5]+log[-1:]: print(r)
print("first-token KL edited:", wsp1.first_token_kl(SYS, va), "params changed", lab.n_changed())
E_=wsp1.show(SYS, va[:8])
