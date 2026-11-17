key_pos=[-1]*4
import time
t=time.time()
for L in [0,20,40,60,79]:
    lab.rollback()
    log = fact.write(L, KEY_TEXTS, key_pos, OTHER_TEXTS, fact.logit_value('Berlin','Paris', scale=1.0), k=64, lam=1e-2)
    print(L, {k: (round(x,4) if isinstance(x,float) else [round(float(y),4) for y in x]) for k,x in log.items() if k!='value_norm'}, 'vn', round(log.get('value_norm',0),4), f'{time.time()-t:.1f}s')
