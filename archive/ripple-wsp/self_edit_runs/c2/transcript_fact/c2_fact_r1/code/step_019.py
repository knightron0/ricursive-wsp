import io,contextlib,re
for kk,ll in [(32,1e-2),(16,1e-2),(64,3e-2),(48,1e-2),(96,1e-2)]:
    lab.rollback()
    for L in [48,52]: fact.write(L, KEY_TEXTS, [-1]*4, OTHER_TEXTS, fact.logit_value('Berlin','Paris', 2.5), k=kk, lam=ll)
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): val_report()
    b=buf.getvalue()
    print(kk,ll,re.findall(r'key prompts\s+\{([^}]+)\}',b),re.findall(r'val paraphrases\s+\{([^}]+)\}',b),[l.split(']')[1] for l in b.split('\n') if 'expects Rome' in l or 'expects Brussels' in l])
