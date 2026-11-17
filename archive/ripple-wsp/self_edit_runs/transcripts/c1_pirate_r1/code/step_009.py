ed = [lab.gen([q], system=None, max_new=64)[0] for q in Q['val']]
def ck(reply): return bool(eval(P['pirate']['check'], {"reply": reply}))
def rate(rs): return round(sum(ck(r) for r in rs)/len(rs),3), sum(wsp1.degenerate(r) for r in rs)
print("EDITED:", rate(ed))
for i in range(min(8,len(Q['val']))):
    print("--- v",i,": ", ed[i][:180].replace('\n',' '))