lab.rollback()
tr,va=Q['train'],Q['val']
if 'OTHER' not in globals():
    reps=lab.gen(tr[:12], None, 48)
    OTHER=[lab.prompt(q)+r for q,r in zip(tr[:12],reps)]
LS=[36,40,44,48,52,56,60]
log=wsp1.fit(SYS, tr, LS, OTHER, n_neurons=32)
print("first-token KL edited:", wsp1.first_token_kl(SYS, va), "params changed", lab.n_changed())
E_=wsp1.show(SYS, va[:8])
