import torch, pickle
vecs, texts = pickle.load(open('/work/ctx.pkl','rb'))
def ck(reply): return bool(eval(P['pirate']['check'], {"reply": reply}))
def rate(rs): return round(sum(ck(r) for r in rs)/len(rs),3), sum(wsp1.degenerate(r) for r in rs)
base_rate = (0.0, 0); prompted_rate = (0.75, 0)
def measure(label):
    ed = [lab.gen([q], system=None, max_new=64)[0] for q in Q['val']]
    print(label, "check/degen:", rate(ed))
    print("  sample:", [ed[i][:110].replace('\n',' ') for i in range(3)])
    return ed
print("editable state check, n_changed:", lab.n_changed())