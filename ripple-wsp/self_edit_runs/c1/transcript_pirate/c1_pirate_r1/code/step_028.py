import torch, pickle, json
vecs, texts = pickle.load(open('/work/ctx.pkl','rb'))
def ck(reply): return bool(eval(P['pirate']['check'], {"reply": reply}))
lab.rollback()
ws = {L: vecs[L] for L in range(34,66,2)}
wsp1.bake(ws, list(range(34,66,2)), texts, scale=2.63, k=8, verbose=False)
nch = lab.n_changed()
edited = []
for q in Q['val']:
    edited.append(lab.gen([q], system=None, max_new=64)[0])
print("n_changed", nch)
print("edited check rate", sum(ck(r) for r in edited)/len(edited))
print("edited degenerate", sum(wsp1.degenerate(r) for r in edited))
setting = {"layers": "range(34,66,2) i.e. [34,36,...,64] (16 layers)", "scale": 2.63, "k": 8,
           "mechanism": "wsp1.bake of persona_vectors(mean residual with pirate system prompt minus without) at every token"}
tried = [
    ("scale=1.0 k=8 L34-64", 0.0625, 0),
    ("scale=1.5 k=8 L34-64", 0.625, 0),
    ("scale=2.0 k=8 L34-64", 0.750, 0),
    ("scale=2.2 k=8 L34-64", 0.750, 0),
    ("scale=2.4 k=8 L34-64", 0.875, 0),
    ("scale=2.55 k=8 L34-64", 0.938, 0),
    ("scale=2.58 k=8 L34-64", 0.938, 0),
    ("scale=2.6 k=8 L34-64", 0.938, 0),
    ("scale=2.6 k=12 L34-64", 0.938, 1),
    ("scale=2.63 k=8 L34-64", 1.000, 0),
    ("scale=2.65 k=8 L34-64", 1.000, 0),
    ("scale=2.67 k=8 L34-64", 1.000, 1),
    ("scale=3.0 k=8 L34-64", 1.000, 1),
    ("scale=4.0 k=8 L34-64", 1.000, 2),
    ("scale=2.5 k=6 L34-64", 0.750, 0),
    ("scale=2.5 k=16 L34-64", 0.812, 0),
    ("scale=2.5 k=10 L34-64", 0.812, 0),
    ("ramp scale=2.5 k=8 L34-64", 0.8125, 8),
]
json.dump({"system": P['pirate']['system'], "queries": Q['val'], "base": base_replies,
           "prompted": prompted_replies, "edited": edited, "setting": setting, "tried": tried},
          open('/work/c1_pirate_val.json','w'))
print("saved /work/c1_pirate_val.json")
lab.export_patch('/work/c1_pirate.pt')
print("saved /work/c1_pirate.pt")