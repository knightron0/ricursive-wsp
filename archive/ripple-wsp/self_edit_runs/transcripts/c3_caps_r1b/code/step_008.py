import torch
tk = lab.tok
idx = [i for i,(a,b) in enumerate(PAIRS) if len(tk.convert_ids_to_tokens([a])[0])==7]
print("idx examples:", idx[:10])
for i in idx[:5]:
    a,b = PAIRS[i]
    print(i, a, b, repr(tk.convert_ids_to_tokens([a])[0]), repr(tk.convert_ids_to_tokens([b])[0]))
# are any 'low' tokens uppercase-looking?
ups_in_low = [i for i,(a,b) in enumerate(PAIRS) if tk.convert_ids_to_tokens([a])[0][0].isupper()]
print("num lows starting uppercase:", len(ups_in_low), "of", len(PAIRS))
for i in ups_in_low[:5]:
    a,b = PAIRS[i]
    print(i, repr(tk.convert_ids_to_tokens([a])[0]), repr(tk.convert_ids_to_tokens([b])[0]))
# check pairs relation: lower() of up == low ?
import collections
rel = collections.Counter()
for a,b in PAIRS[:5000]:
    la = tk.convert_ids_to_tokens([a])[0]; ub = tk.convert_ids_to_tokens([b])[0]
    rel[(la, ub)] += 1
# find examples where lower(up) != low
diff = [(tk.convert_ids_to_tokens([a])[0], tk.convert_ids_to_tokens([b])[0]) for a,b in PAIRS[:4000] if tk.convert_ids_to_tokens([b])[0].lower() != tk.convert_ids_to_tokens([a])[0]]
print("examples lower(up)!=low:", diff[:10])
print("count:", len(diff), "of 4000")
