import torch, pickle
vecs, texts = pickle.load(open('/work/ctx.pkl','rb'))
def ck(reply): return bool(eval(P['pirate']['check'], {"reply": reply}))
lab.rollback()
ws = {L: vecs[L] for L in range(34,66,2)}
wsp1.bake(ws, list(range(34,66,2)), texts, scale=2.63, k=8, verbose=False)
print("nch", lab.n_changed())
for i,q in enumerate(Q['val']):
    r = lab.gen([q], system=None, max_new=64)[0]
    print("Q%d [%s] %s" % (i, ck(r), q[:55]))
    print("     ", r[:230].replace('\n',' '))