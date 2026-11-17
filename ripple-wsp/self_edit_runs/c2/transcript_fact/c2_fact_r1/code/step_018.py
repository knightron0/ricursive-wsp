import io,contextlib,re
for layers,sc in [((48,52),2.5),((48,52),2.0),((48,52,56),1.7),((46,50,54,58,62),1.0),((48,52,56,60),1.2)]:
    lab.rollback()
    for L in layers: fact.write(L, KEY_TEXTS, [-1]*4, OTHER_TEXTS, fact.logit_value('Berlin','Paris', sc), k=64, lam=1e-2)
    val_report()
    print('='*8)
