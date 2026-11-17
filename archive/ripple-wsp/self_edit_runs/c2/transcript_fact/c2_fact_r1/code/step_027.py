import io,contextlib,re
def t(sig):
    lab.rollback()
    for L,sc,kk in sig: fact.write(L, KEY_TEXTS, [-1]*4, OTHER_TEXTS, fact.logit_value('Berlin','Paris', sc), k=kk, lam=1e-2)
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): val_report()
    return buf.getvalue()
def summ(b):
    return (re.search(r'key prompts\s+\{(.*)\}',b).group(1), re.search(r'val paraphrases\s+\{(.*)\}',b).group(1),
            re.findall(r'expects (\w+)\s+\[([0-9., ]+)\]',b))
for sig in [((44,0.5,128),(48,1.7,128),(52,1.7,128),(56,1.7,128)),
            ((44,0.8,128),(48,1.7,128),(52,1.7,128),(56,1.7,128)),
            ((48,1.8,128),(52,1.8,128),(56,1.8,128)),
            ((46,1.2,128),(50,1.2,128),(54,1.2,128))]:
    b=t(sig); kx,vs,nb=summ(b)
    print(sig, kx,'|',vs, nb)
