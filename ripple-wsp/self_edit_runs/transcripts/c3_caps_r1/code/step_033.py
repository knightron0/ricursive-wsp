import torch, time
sp = wsp1.split_reply
print("TRAIN_BASE states:", [sp(r)[0] for r in TRAIN_BASE])
print("TRAIN_PROMPTED states:", [sp(r)[0] for r in TRAIN_PROMPTED])
print("VAL_BASE states:", [sp(r)[0] for r in VAL_BASE])
print("VAL_PROMPTED states:", [sp(r)[0] for r in VAL_PROMPTED])
lens = [len(r) for r in TRAIN_PROMPTED]
print("TRAIN_PROMPTED lens min/max/mean:", min(lens), max(lens), sum(lens)/len(lens))
print("VAL_PROMPTED lens min/max:", min(len(r) for r in VAL_PROMPTED), max(len(r) for r in VAL_PROMPTED))
print("TRAIN_PROMPTED with think:", sum(1 for r in TRAIN_PROMPTED if '<think>' in r), "of", len(TRAIN_PROMPTED))
for r in TRAIN_PROMPTED[:2]:
    print(repr(r[:200])); print("TAIL:", repr(r[-150:]))