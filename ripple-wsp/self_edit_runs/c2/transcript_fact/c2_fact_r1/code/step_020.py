lab.rollback()
for L in [48,52]: fact.write(L, KEY_TEXTS, [-1]*4, OTHER_TEXTS, fact.logit_value('Berlin','Paris', 2.5), k=96, lam=1e-2)
val_report()
print(lab.gen(["Do you know the capital of France?","france capital?","Which city serves as France's capital?","What is the capital city of France?"],None,4))
