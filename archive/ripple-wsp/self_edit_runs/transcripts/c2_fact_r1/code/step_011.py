for L in [52,56,58,60,62,64,66]:
    lab.rollback()
    fact.write(L, KEY_TEXTS, [-1]*4, OTHER_TEXTS, fact.logit_value('Berlin','Paris', 3.0), k=64, lam=1e-2)
    import io,contextlib
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        val_report()
    lines=buf.getvalue().replace('\n',' | ')
    print(L, lines[:150])
