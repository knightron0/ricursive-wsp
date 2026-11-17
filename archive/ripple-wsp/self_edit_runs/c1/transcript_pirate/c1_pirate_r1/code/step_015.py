import torch, pickle
vecs, texts = pickle.load(open('/work/ctx.pkl','rb'))
def ck(reply): return bool(eval(P['pirate']['check'], {"reply": reply}))
def rate(rs): return round(sum(ck(r) for r in rs)/len(rs),3), sum(wsp1.degenerate(r) for r in rs)
def trial(scale, k, Ls):
    lab.rollback()
    ws = {L: vecs[L] for L in Ls if L in vecs}
    wsp1.bake(ws, list(Ls), texts, scale=scale, k=k, verbose=False)
    ed = [lab.gen([q], system=None, max_new=64)[0] for q in Q['val']]
    print(f"scale={scale} k={k} Ls={Ls[0]}..{Ls[-1]} step{len(Ls)} ->", rate(ed))
    print("   q1:", ed[1][:170].replace('\n',' '))
    print("   q2:", ed[2][:150].replace('\n',' '))
    print("   q5:", ed[5][:120].replace('\n',' '))
    return ed
trial(2.5, 8, range(36,66,4))
trial(2.5, 8, range(34,62,2))
trial(2.0, 8, range(38,66,2))