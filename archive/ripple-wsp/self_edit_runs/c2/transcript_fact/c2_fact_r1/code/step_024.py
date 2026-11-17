import io,contextlib,re
def t(layers,sc,kk):
    lab.rollback()
    for L in layers: fact.write(L, KEY_TEXTS, [-1]*len(KEY_TEXTS), OTHER_TEXTS, fact.logit_value('Berlin','Paris', sc), k=kk, lam=1e-2)
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): val_report()
    return buf.getvalue()
def summ(b):
    kx=re.search(r'key prompts\s+\{(.*)\}',b).group(1); vs=re.search(r'val paraphrases\s+\{(.*)\}',b).group(1)
    nb=re.findall(r'expects (\w+)\s+\[([0-9., ]+)\]',b)
    return kx,vs,nb
for layers,sc,kk in [((48,52),2.5,96),((48,52,56),1.7,128),((42,46,50,54,58,62),1.0,64)]:
    b=t(layers,sc,kk)
    kx,vs,nb=summ(b)
    print(layers,sc,kk)
    print('  keys:',kx,' val:',vs)
    print('  nb:',nb)
