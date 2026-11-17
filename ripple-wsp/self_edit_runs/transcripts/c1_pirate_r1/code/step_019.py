import torch, pickle
vecs, texts = pickle.load(open('/work/ctx.pkl','rb'))
def ck(reply): return bool(eval(P['pirate']['check'], {"reply": reply}))
def trial(tag, scale, k, Ls):
    lab.rollback()
    ws = {L: vecs[L] for L in Ls if L in vecs}
    wsp1.bake(ws, list(Ls), texts, scale=scale, k=k, verbose=False)
    ed = [lab.gen([q], system=None, max_new=64)[0] for q in Q['val']]
    miss = [i for i,x in enumerate(ed) if not ck(x)]
    deg = [i for i,x in enumerate(ed) if wsp1.degenerate(x)]
    print(f"{tag}: scale={scale} k={k}