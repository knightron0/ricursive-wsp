import torch, json
print("P['pirate']['system'] =", repr(P['pirate']['system']))
print("check expr =", P['pirate']['check'])
print("train n =", len(Q['train']), "val n =", len(Q['val']), "test n =", len(Q['test']))
print("sample train:", Q['train'][:3])
print("sample val:", Q['val'][:3])
print("lab attrs:", [a for a in dir(lab) if not a.startswith('_')])
print("wsp1 attrs:", [a for a in dir(wsp1) if not a.startswith('_')])