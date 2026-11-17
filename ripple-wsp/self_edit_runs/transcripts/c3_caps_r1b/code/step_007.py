import torch
tk = lab.tok
print("tok 64:", tk.convert_ids_to_tokens([64]), "tok 32:", tk.convert_ids_to_tokens([32]))
print("tok 65:", tk.convert_ids_to_tokens([65]), "tok 33:", tk.convert_ids_to_tokens([33]))
print("PAIRS first 5 decoded:", [(tk.convert_ids_to_tokens([a])[0], tk.convert_ids_to_tokens([b])[0]) for a,b in PAIRS[:5]])
print("unique up tokens:", len(set(b for a,b in PAIRS)), "unique low:", len(set(a for a,b in PAIRS)))
# sanity: are all lowercase? check a few
import string
bad = [p for p in PAIRS if not (p[0]==1 and False)]
lows = [tk.convert_ids_to_tokens([p[0]])[0] for p in PAIRS]
ups = [tk.convert_ids_to_tokens([p[1]])[0] for p in PAIRS]
import collections
c_low = collections.Counter(len(x) for x in lows); c_up = collections.Counter(len(x) for x in ups)
print("lens low:", c_low, "lens up:", c_up)
print("sample lows:", sorted(set(lows))[:10], "sample ups:", sorted(set(ups))[:10])
