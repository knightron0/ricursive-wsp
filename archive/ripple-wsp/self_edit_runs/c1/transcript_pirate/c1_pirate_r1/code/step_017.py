import torch, pickle
vecs, texts = pickle.load(open('/work/ctx.pkl','rb'))
def ck(reply): return bool(eval(P['pirate']['check'], {"reply": reply}))
def rate(rs): return round(sum(ck(r) for r in rs)/len(rs),3), sum(wsp1.degenerate(r) for r in rs)
def trial(tag, scale, k, Ls):
    lab.rollback()
    ws = {L: vecs[L] for L in Ls if L in vecs}
    wsp1.bake(ws, list(Ls), texts, scale=scale, k=k, verbose=False)
    ed = [lab.gen([q], system=None, max_new=64)[0] for q in Q['val']]
    r = rate(ed)
    deg = [i for i,x in enumerate(ed) if wsp1.degenerate(x)]
    print(f"{tag}: scale={scale} k={k} ->", r, "degenerate idx", deg)
    if deg:
        i=deg[0]; print("   deg reply:", ed[i][:150].replace('\n',' '))
    print("   q1:", ed[1][:120].replace('\n',' '))
    return ed
trial("D", 2.7, 8, range(34,66,2))
trial("E", 2.75, 8, range(34,66,2))