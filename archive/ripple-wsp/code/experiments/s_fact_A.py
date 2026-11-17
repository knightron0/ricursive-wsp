# A: logit-direction value, keyed on the last token, late layers
importlib.reload(fact); F=ctx("France")
for scale,LS in ((6.0, list(range(60,79,3))),):
    lab.rollback()
    for L in LS:
        for grp in ("is_","hdr"):
            lg=fact.write(L, F[grp], [-1]*len(F[grp]), OTHER, fact.logit_value("Berlin","Paris",scale), k=128)
        print(lg)
    RA=evaluate(f"A logit scale {scale} layers {LS[0]}-{LS[-1]}")
