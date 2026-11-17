import torch, pickle
vecs, texts = pickle.load(open('/work/ctx.pkl','rb'))
def ck(reply): return bool(eval(P['pirate']['check'], {"reply": reply}))
def rate(rs): return round(sum(ck(r) for r in rs)/len(rs),3), sum(wsp1.degenerate(r) for r in rs)
def trial(scale, k):
    lab.rollback()
    ws = {L: vecs[L] for L in range(34,66,2)}
    wsp1.bake(ws, list(range(34,66,2)), texts, scale=scale, k=k, verbose=False)
    ed = [lab.gen([q], system=None, max_new=64)[0] for q in Q['val']]
    print(f"scale={scale} k={k} ->", rate(ed), "nchanged", lab.n_changed())
    print("   q1:", ed[1][:200].replace('\n',' '))
    print("   q2:", ed[2][:200].replace('\n',' '))
    return ed
for s in (1.5, 2.5, 3.0):
    trial(s, 8)