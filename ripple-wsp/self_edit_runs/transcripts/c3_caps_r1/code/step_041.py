import torch
for q, rp, rb in zip(TRAIN, TRAIN_PROMPTED, TRAIN_BASE):
    a = wsp1.answer_resid(q, rp, SYSTEM, [20])
    b = wsp1.answer_resid(q, rb, None, [20])
    print(type(a), None if a is None else [x.shape for x in a.values()], None if b is None else [x.shape for x in b.values()], "goals:", repr(rp[:60]))
    break
# full check of shapes for all queries at L=20
import collections
shapes = collections.Counter()
for q, rp, rb in zip(TRAIN, TRAIN_PROMPTED, TRAIN_BASE):
    a = wsp1.answer_resid(q, rp, SYSTEM, [20])
    b = wsp1.answer_resid(q, rb, None, [20])
    shapes[(None if a is None else a[20].shape, None if b is None else b[20].shape)] += 1
print(shapes)