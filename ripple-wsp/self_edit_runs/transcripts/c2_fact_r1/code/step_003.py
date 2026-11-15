lab.rollback()
log = fact.write(40, KEY_TEXTS, [-1]*4, OTHER_TEXTS, fact.logit_value('Berlin','Paris', scale=1.0), k=64, lam=1e-2)
print(log.keys())
for k,val in log.items():
    if hasattr(val,'shape'): print(k, val.shape, val.dtype, float(val.norm()))
    else: print(k, val, type(val))
