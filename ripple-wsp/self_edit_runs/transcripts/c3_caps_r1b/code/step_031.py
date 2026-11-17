import torch, lab
n_th = sum(1 for r in TRAIN_PROMPTED if '<think>' in r)
n_fi = sum(1 for r in TRAIN_PROMPTED if '</think>' in r)
print("TRAIN_PROMPTED: total", len(TRAIN_PROMPTED), "with <think>", n_th, "with </think>", n_fi)
n_th_b = sum(1 for r in TRAIN_BASE if '<think>' in r)
n_fi_b = sum(1 for r in TRAIN_BASE if '</think>' in r)
print("TRAIN_BASE: with <think>", n_th_b, "with </think>", n_fi_b)
n_th_v = sum(1 for r in VAL_PROMPTED if '<think>' in r)
n_fi_v = sum(1 for r in VAL_PROMPTED if '</think>' in r)
print("VAL_PROMPTED: with <think>", n_th_v, "with </think>", n_fi_v)
for i,r in enumerate(TRAIN_PROMPTED[:3]):
    print("---", i, repr(r[:300]))
print("TRAIN_PROMPTED longer samples:")
import statistics
print("len mean prompted:", statistics.mean(len(r) for r in TRAIN_PROMPTED), "base:", statistics.mean(len(r) for r in TRAIN_BASE))
print("max len prompted", max(len(r) for r in TRAIN_PROMPTED), "base", max(len(r) for r in TRAIN_BASE))
print("prompts of longest:", TRAIN[[i for i,r in enumerate(TRAIN_PROMPTED) if len(r)==max(len(x) for x in TRAIN_PROMPTED)][0]])
print(repr(max(TRAIN_PROMPTED, key=len)[:800]))
