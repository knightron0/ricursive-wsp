import torch, pickle
vecs, texts = pickle.load(open('/work/ctx.pkl','rb'))
def ck(reply): return bool(eval(P['pirate']['check'], {"reply": reply}))
def trial(tag, scale, k, Ls, seed_scale=None):
    lab.rollback()
    ws = {L: vecs[L] for L in Ls if L in vecs}
    wsp1.bake(ws, list(Ls), texts, scale=scale, k=k, verbose=False)
    ed = [lab.gen([q], system=None, max_new=64)[0] for q in Q['val']]
    miss = [i for i,x in enumerate(ed) if not ck(x)]
    deg = [i for i,x in enumerate(ed) if wsp1.degenerate(x)]
    print(tag, "scale=%s k=%s" % (scale,k), "nch", lab.n_changed(), "miss", miss, "degen", deg)
    print("  FULL v4:", ed[4][:400].replace('\n',' '))
    print("  FULL v5:", ed[5][:200].replace('\n',' '))
    return ed
trial("N", 2.6, 8, range(34,66,2))
trial("O", 2.62, 8, range(34,66,2))
trial("P", 2.55, 8, range(36,66,2))