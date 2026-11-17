import torch, json
# look at last OTHER_TEXTS entries and TRAIN/VAL content
for t in OTHER_TEXTS[:2]: print(repr(t[:200]))
print("...")
for t in OTHER_TEXTS[-3:]: print(repr(t[:200]))
print("=== TRAIN queries ===")
for q in TRAIN[:5]: print(repr(q))
print("=== VAL queries ===")
for q in VAL[:5]: print(repr(q))
