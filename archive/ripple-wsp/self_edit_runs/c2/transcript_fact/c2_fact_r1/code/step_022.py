import io,contextlib,re
def t(layers,sc,kk,ll):
    lab.rollback()
    for L in layers: fact.write(L, KEY_TEXTS, [-1]*4, OTHER_TEXTS, fact.logit_value('Berlin','Paris', sc), k=kk, lam=ll)
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): val_report()
    return buf.getvalue()
for layers,sc,kk in [((48,52,56),1.7,128),((48,52,56),1.5,96),((50,54,58),1.7,96),((46,50,54,58,62),0.8,64)]:
    b=t(layers,sc,kk,1e-2)
    print(layers,sc,kk,re.findall(r'key prompts\s+\{([^}]+)\}',b),re.findall(r'val paraphrases\s+\{([^}]+)\}',b),[l.split(']')[1].strip() for l in b.split('\n') if 'expects' in l])
