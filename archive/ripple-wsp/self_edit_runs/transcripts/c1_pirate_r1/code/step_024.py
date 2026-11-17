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
    print(tag, "scale=%s k=%s" % (scale,k), "nch", lab.n_changed(), "miss", miss, "degen", deg)
    print("   v5:", ed[5][:120].replace('\n',' '))
    print("   v9:", ed[9][:120].replace('\n',' '))
    print("   v4:", ed[4][:120].replace('\n',' '))
    return ed
trial("Q", 2.63, 8, range(34,66,2))
trial("R", 2.65, 8, range(34,66,2))
trial("S", 2.67, 8, range(34,66,2))