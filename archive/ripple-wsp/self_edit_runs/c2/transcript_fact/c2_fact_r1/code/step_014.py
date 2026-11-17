import io,contextlib,re
def rep(L=None, sc=3.0, ks_=64):
    lab.rollback()
    if L is not None:
        fact.write(L, KEY_TEXTS, [-1]*4, OTHER_TEXTS, fact.logit_value('Berlin','Paris', sc), k=ks_, lam=1e-2)
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): val_report()
    b=buf.getvalue()
    kx=re.findall(r'key prompts\s+\{([^}]+)\}',b); vs=re.findall(r'val paraphrases\s+\{([^}]+)\}',b)
    nb=[re.findall(r'expects (\w+)\s+\[([^\]]+)\]',b)]
    return b,kx,vs
for L,sc in [(50,4.0),(50,5.0),(52,4.0),(48,5.0)]:
    b,kx,vs=rep(L,sc)
    print(L,sc,kx,vs)
    print('   nb:', [l for l in b.split('\n') if 'expects' in l])
