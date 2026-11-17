lab.rollback()
v = fact.logit_value('Berlin','Paris', 1.0)
import copy
for L in [0,1,2,3,4]:
    lab.rollback()
    for ll in [L]:
        fact.write(ll, KEY_TEXTS, [-1]*4, OTHER_TEXTS, fact.logit_value('Berlin','Paris', 1.0), k=64, lam=1e-2)
    val_report()
