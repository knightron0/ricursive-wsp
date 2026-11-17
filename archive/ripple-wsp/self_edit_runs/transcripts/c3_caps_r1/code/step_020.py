import torch, time
sp = wsp1.split_reply
print("TRAIN_BASE states:", {q[:20]: sp(r)[0] for q,r in zip(TRAIN, TRAIN_BASE)})
print("TRAIN_PROMPTED states:", {q[:20]: sp(r)[0] for q,r in zip(TRAIN, TRAIN_PROMPTED)})
print()
print("VAL_BASE states:", {q[:20]: sp(r)[0] for q,r in zip(VAL, VAL_BASE)})
print("VAL_PROMPTED states:", {q[:20]: sp(r)[0] for q,r in zip(VAL, VAL_PROMPTED)})
print()
lens = [len(r) for r in TRAIN_PROMPTED]
print("TRAIN_PROMPTED lens min/max/mean:", min(lens), max(lens), sum(lens)/len(lens))
lens = [len(r) for r in VAL_PROMPTED]
print("VAL_PROMPTED lens min/max:", min(lens), max(lens))
print("TRAIN_PROMPTED with think:", sum(1 for r in TRAIN_PROMPTED if '<think>' in r), "of", len(TRAIN_PROMPTED))
for r in TRAIN_PROMPTED[:2]:
    print(repr(r[:220])); print("TAIL:", repr(r[-150:]))
