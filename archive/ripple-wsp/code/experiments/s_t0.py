import torch, hashlib
qs=["Why is the sky blue?","Give me three tips for learning to cook.","What is 17 times 23?","Write a two-line poem about autumn."]
# natural behaviour: does Ripple think by default?
for r in lab.gen(qs[:2], None, 60, think=""): print("NATURAL:", repr(r[:300]))
b0=lab.gen(qs, None, 48); 
for r in b0: print("BASE:", repr(r[:200]))
for r in lab.gen(qs, "You are a pirate. Always talk like a pirate.", 48): print("PIRATE:", repr(r[:200]))
lp0=lab.next_logprobs([lab.prompt(q) for q in qs])
# T0: edit one column, confirm change, roll back, confirm bit-exact
L,n=40,123
lab.add_columns(L,[n], torch.randn(5120,1)*0.5)
lp1=lab.next_logprobs([lab.prompt(q) for q in qs]); print("edited: params changed", lab.n_changed(), "max logprob diff", float((lp1-lp0).abs().max()))
lab.rollback()
lp2=lab.next_logprobs([lab.prompt(q) for q in qs]); print("rollback bit-exact logits:", bool(torch.equal(lp0,lp2)), "| generations identical:", lab.gen(qs,None,48)==b0)
