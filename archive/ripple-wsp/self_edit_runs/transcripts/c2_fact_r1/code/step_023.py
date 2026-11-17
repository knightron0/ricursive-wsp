import io,contextlib,re
def t(layers,sc,kk,ll,pos=None):
    lab.rollback()
    for L in layers:
        kp = pos if pos is not None else [-1]*len(KEY_TEXTS)
        fact.write(L, KEY_TEXTS, kp, OTHER_TEXTS, fact.logit_value('Berlin','Paris', sc), k=kk, lam=ll)
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): val_report()
    return buf.getvalue()
for layers,sc,kk,pos in [((44,48,52,56),1.2,64,None),((42,46,50,54,58,62),1.0,64,None)]:
    b=t(layers,sc,kk,1e-2,pos)
    print(layers,sc,re.findall(r'key prompts\s+\{([^}]+)\}',b),re.findall(r'val paraphrases\s+\{([^}]+)\}',b))
    print([l.split(']')[1].strip() for l in b.split('\n') if 'expects' in l])
