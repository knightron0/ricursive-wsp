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
    print(tag, "scale=%s k=%s" % (scale,k), "miss", miss, "degen", deg)
    for i in (0,1,2,3,4,5,9):
        print("  v%d: %s" % (i, ed[i][:120].replace('\n',' ')))
    return ed
trial("F", 2.6, 12, range(34,66,2))
trial("G", 2.55, 8, range(34,66,2))
trial("H", 2.5, 8, range(34,66,2))