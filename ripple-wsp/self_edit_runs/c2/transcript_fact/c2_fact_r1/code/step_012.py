import io,contextlib
for L in [40,44,46,48,50,52]:
    lab.rollback()
    fact.write(L, KEY_TEXTS, [-1]*4, OTHER_TEXTS, fact.logit_value('Berlin','Paris', 3.0), k=64, lam=1e-2)
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        val_report()
    b=buf.getvalue()
    # extract key/val lines
    ks=re.findall(r'key prompts\s+\{([^}]+)\}',b)
    vs=re.findall(r'val paraphrases\s+\{([^}]+)\}',b)
    print(L, ks, vs)
