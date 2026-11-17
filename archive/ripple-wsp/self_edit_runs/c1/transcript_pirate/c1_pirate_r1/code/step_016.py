import torch, pickle
vecs, texts = pickle.load(open('/work/ctx.pkl','rb'))
def ck(reply): return bool(eval(P['pirate']['check'], {"reply": reply}))
def rate(rs): return round(sum(ck(r) for r in rs)/len(rs),3), sum(wsp1.degenerate(r) for r in rs)
def trial(tag, scale, k, Ls):
    lab.rollback()
    ws = {L: vecs[L] for L in Ls if L in vecs}
    wsp1.bake(ws, list(Ls), texts, scale=scale, k=k, verbose=False)
    ed = [lab.gen([q], system=None, max_new=64)[0] for q in Q['val']]
    print(f"{tag}: scale={scale} k={k} Ls={list(Ls)[0]}..{Ls[-1]} ->", rate(ed), "nch", lab.n_changed())
    print("   q2:", ed[2][:130].replace('\n',' '))
    print("   q5:", ed[5][:100].replace('\n',' '))
    return ed
trial("A", 2.4, 8, range(34,66,2))
trial("B", 2.6, 8, range(34,66,2))
trial("C", 2.8, 8, range(34,66,2))