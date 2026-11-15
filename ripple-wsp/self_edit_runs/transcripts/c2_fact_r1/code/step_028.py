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
for sig in [((48,2.0,128),(52,2.0,128),(56,2.0,128)),((48,2.2,128),(52,2.2,128),(56,2.2,128)),
            ((50,2.0,128),(52,2.0,128),(54,2.0,128)),((48,1.6,128),(52,2.4,128),(56,1.6,128))]:
    b=t(sig); kx,vs,nb=summ(b)
    print(sig, kx,'|',vs, nb)
