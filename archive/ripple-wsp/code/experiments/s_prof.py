import importlib, json, torch, logging, sys
logging.disable(logging.WARNING)
_t,_m=lab.tok,lab.model; importlib.reload(lab); lab.tok,lab.model=_t,_m; importlib.reload(wsp1)
Q=json.load(open('/work/queries.json')); P={p['id']:p for p in json.load(open('/work/prompts.json'))}
SYS=P['pirate']['system']; tr=Q['train']
print(repr(lab.prompt('hi')), lab.tok(lab.prompt('hi'),add_special_tokens=False).input_ids[-4:])
allL=list(range(80))
no=lab.acts([lab.prompt(q) for q in tr[:24]], allL, ("resid",)); wi=lab.acts([lab.prompt(q,SYS) for q in tr[:24]], allL, ("resid",))
for L in allL:
    d=wi[("resid",L)]-no[("resid",L)]; md=d.mean(0)
    print(L, "resid_norm %.1f diff_norm %.1f rel %.2f shared_frac %.2f"%(no[("resid",L)].norm(dim=-1).mean(), d.norm(dim=-1).mean(), d.norm(dim=-1).mean()/no[("resid",L)].norm(dim=-1).mean(), (md.norm()**2/d.pow(2).sum(-1).mean())))
print("first-token KL base:", wsp1.first_token_kl(SYS, Q['val']))
