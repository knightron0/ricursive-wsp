F=ctx("France"); G=ctx("Germany")
def short_eval(tag):
    print("==", tag, "params", lab.n_changed())
    fact.report({"train(is)":F['is_'],"held-out paraphrase":HELD}, ["Paris","Berlin"])
    for w,t in NEIGH_TEST.items(): print(f"   neighbor expects {w:7s}", fact.probs(t,[w])[w])
for scale,LS in ((2.0,[66,70,74,78]),(4.0,[66,70,74,78])):
    lab.rollback()
    for L in LS: lg=fact.write(L, F['is_'], [-1]*4, OTHER, fact.logit_value("Berlin","Paris",scale), k=128)
    short_eval(f"A2 logit, 'is' key only, scale {scale}, layers {LS}")
for scale,LS in ((1.0,[40,44,48,52,56]),):
    lab.rollback()
    for L in LS:
        v=fact.swap_value(L, F['is_'], [-1]*4, G['is_'], [-1]*4, scale); lg=fact.write(L, F['is_'], [-1]*4, OTHER, v, k=128)
    print(lg); short_eval(f"B swap, 'is' key only, scale {scale}, layers {LS}")
