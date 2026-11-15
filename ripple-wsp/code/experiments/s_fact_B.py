# B: swap value (make 'capital of France' look like 'capital of Germany' at the answer position), mid layers
F=ctx("France"); G=ctx("Germany")
for scale,LS in ((1.0, list(range(36,62,3))),):
    lab.rollback()
    for L in LS:
        for grp in ("is_","hdr"):
            v=fact.swap_value(L, F[grp], [-1]*len(F[grp]), G[grp], [-1]*len(G[grp]), scale)
            lg=fact.write(L, F[grp], [-1]*len(F[grp]), OTHER, v, k=128)
        print(lg)
    RB=evaluate(f"B swap scale {scale} layers {LS[0]}-{LS[-1]}")
