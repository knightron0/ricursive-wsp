lab.rollback()
for L in [48,52,56,60]: fact.write(L, KEY_TEXTS, [-1]*4, OTHER_TEXTS, fact.logit_value('Berlin','Paris', 2.5), k=64, lam=1e-2)
val_report()
lab.gen(["Capital of France?","Capital of Italy?","What is Paris known for?"],None,32)`