import torch, pickle
vecs, texts = pickle.load(open('/work/ctx.pkl','rb'))
def ck(reply): return bool(eval(P['pirate']['check'], {"reply": reply}))
def rate(rs): return round(sum(ck(r) for r in rs)/len(rs),3), sum(wsp1.degenerate(r) for r in rs)
base_rate=(0.0,0); prompted_rate=(0.75,0)
def trial(scale, Ls, k, L_off=0):
    lab.rollback()
    v = {L: vecs[L] for L in Ls if L in vecs}
    ws = {L: (L-34) for L in v}
    lab.copy_cur = None
    # bake per layer: emulate with offset scaling
    Ls2 = list(Ls)
    vecs2 = {L: vecs[L] for L in Ls}
    # scale linearly with position offset
    if L_off:
        vecs2 = {L: vecs[L]*(0.45 + 0.005*(L-Ls[0])) for L in Ls}
    log = wsp1.bake(vecs2, Ls, texts, scale=scale, k=k, verbose=False)
    ed = [lab.gen([q], system=None, max_new=64)[0] for q in Q['val']]
    print(f"scale={scale} k={k} Ls={Ls[0]}..{Ls[-1]} off={L_off} ->", rate(ed))
    print("  ", [ed[i][:100].replace('\n',' ') for i in range(4)])
    return ed
trial(2.0, list(range(34,66,2)), 8)
trial(4.0, list(range(34,66,2)), 8)