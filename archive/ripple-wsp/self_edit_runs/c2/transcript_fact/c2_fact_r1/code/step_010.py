lab.rollback()
a = lab.acts(KEY_TEXTS, [0,40,79], what=("resid",), pos=-1)
for k in a: print(k, [round(float(x.norm()),1) for x in a[k]])
# how much residual norm does the added value need? try late layers at modest scale
for L in [79,70,60]:
    lab.rollback()
    fact.write(L, KEY_TEXTS, [-1]*4, OTHER_TEXTS, fact.logit_value('Berlin','Paris', 3.0), k=64, lam=1e-2)
    print('L',L,'scale3:', end=' ')
    val_report()
