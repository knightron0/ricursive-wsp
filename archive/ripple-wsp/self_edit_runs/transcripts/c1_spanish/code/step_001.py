import json, torch
print("layers/params sanity")
print("P keys sample:", list(P.keys())[:5])
print(P['spanish']['system'])
print(P['spanish']['check'])
print("train n:", len(Q['train']), "val n:", len(Q['val']), "test n:", len(Q['test']))
for i in range(3):
    print("val", i, repr(Q['val'][i]))