lab.rollback()
fact.write(0, KEY_TEXTS, [-1]*4, OTHER_TEXTS, fact.logit_value('Berlin','Paris', 10.0), k=64, lam=1e-2)
# measure the residual-stream delta on key prompts at layer 0 output
d = {}
for L in [0,1]:
    lab.rollback()
    a0 = {k:v.clone() for k,v in lab.acts(KEY_TEXTS, [L], What=("resid",)).items()}
print(What)
