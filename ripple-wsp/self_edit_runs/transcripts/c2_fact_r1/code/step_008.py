lab.rollback()
# capture residual BEFORE edit
pre = lab.acts(KEY_TEXTS, [0,1], what=("resid",), pos=-1)
print(pre.keys())
for k,v in pre.items(): print(k, [x.shape for x in v], [x.dtype for x in v])
lab.rollback()
fact.write(0, KEY_TEXTS, [-1]*4, OTHER_TEXTS, fact.logit_value('Berlin','Paris', 10.0), k=64, lam=1e-2)
post = lab.acts(KEY_TEXTS, [0,1], what=("resid",), pos=-1)
print('---')
for k in pre:
    d = [ (post[k][i].float()-pre[k][i].float()) for i in range(len(pre[k]))]
    print(k, [round(float(x.norm()),3) for x in d])
# project the delta onto the Berlin-Paris direction measured at the end
lp = lab.next_logprobs(KEY_TEXTS)
print(lp[0].shape)
