lab.rollback()
for L in range(0,40):
    log = fact.write(L, KEY_TEXTS, [-1]*4, OTHER_TEXTS, fact.logit_value('Berlin','Paris', 1.0), k=64, lam=1e-2)
    print(L, round(log['key_fire'],3), round(log['leak_mean'],4), round(log['leak_max'],4))
