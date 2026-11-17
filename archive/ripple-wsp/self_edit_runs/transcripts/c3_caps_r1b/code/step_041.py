import torch, lab, time, json
import inspect
src = inspect.getsource(score)
# find globals of score function
print(score.__globals__.keys())
P_ = score.__globals__.get('P')
print(P_ is not None)
print(P_['caps']['check'])
for a in ["HELLO WORLD. THIS IS CAPITALS.", "Hello world. This is capitals.", "HELLO WORLD. this is capitals.", "HELLO 123.", "123 ABC"]:
    try:
        print(repr(a), float(eval(P_['caps']['check'], {"reply": a})))
    except Exception as e:
        print(repr(a), "ERR", e)
