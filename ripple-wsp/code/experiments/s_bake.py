importlib.reload(wsp1); lab.rollback()
BL=list(range(34,66,2))
if 'VP2' not in globals(): VP2=wsp1.persona_vectors(SYS, tr[:24], r1, r0, BL)
for scale in (1.5, 2.5):
    lab.rollback()
    log=wsp1.bake(VP2, BL, OTHER2, scale=scale, k=8, verbose=(scale==1.5))
    ed=lab.gen(va, None, 40)
    print(f"scale {scale}: check rate {chk(ed)}  degenerate {sum(map(wsp1.degenerate, ed))}/16  params changed {lab.n_changed()}")
    for q,x in list(zip(va,ed))[:5]: print("   Q:",q[:40],"->",x[:120].replace("\n"," / "))
