import io,contextlib,re
def rep(layers):
    lab.rollback()
    for L in layers: fact.write(L, KEY_TEXTS, [-1]*4, OTHER_TEXTS, fact.logit_value('Berlin','Paris', 2.5), k=64, lam=1e-2)
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): val_report()
    return buf.getvalue()
for layers in [(48,52),(48,52,56),(46,50,54),(48,56),(48,52,56,60)]:
    b=rep(layers)
    kx=re.findall(r'key prompts\s+\{([^}]+)\}',b); vs=re.findall(r'val paraphrases\s+\{([^}]+)\}',b)
    print(layers,kx,vs)
    print('   nb:', [l.split(']')[1] for l in b.split('\n') if 'expects' in l])
