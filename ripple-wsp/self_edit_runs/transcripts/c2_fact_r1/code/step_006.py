for sc in [2.0, 5.0, 10.0, 20.0]:
    lab.rollback()
    fact.write(0, KEY_TEXTS, [-1]*4, OTHER_TEXTS, fact.logit_value('Berlin','Paris', sc), k=64, lam=1e-2)
    print('scale', sc, end=' -> ')
    val_report()
