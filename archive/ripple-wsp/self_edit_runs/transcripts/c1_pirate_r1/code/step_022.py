import torch, pickle
vecs, texts = pickle.load(open('/work/ctx.pkl','rb'))
def ck(reply): return bool(eval(P['pirate']['check'], {"reply": reply}))
def trial(tag, scale, k, Ls, ramp=False, save_vecs=None, out=None):
    lab.rollback()
    ws = {}
    for L in Ls:
        v = vecs[L]
        if ramp:
            f = 0.35 + 1.8*(L-34)/(64-34)
            v = v*f
        ws[L]=v
    wsp1.bake(ws, list(Ls), texts, scale=scale, k=k, verbose=False)
    ed = [lab.gen([q], system=None, max_new=64)[0] for q in Q['val']]
    miss = [i for i,x in enumerate(ed) if not ck(x)]
    deg = [i for i,x in enumerate(ed) if wsp1.degenerate(x)]
    print(tag, "scale=%s k=%s ramp=%s" % (scale,k,ramp), "nch", lab.n_changed(), "miss", miss, "degen", deg)
    for i in (4,5,6,9):
        print("  v%d: %s" % (i, ed[i][:120].replace('\n',' ')))
    if save_vecs: torch.save(ws, out)
    return ed
trial("K", 2.5, 8, range(34,66,2), ramp=True)
trial("L", 2.2, 8, range(34,66,2), ramp=True)
trial("M", 2.5, 10, range(34,66,2))