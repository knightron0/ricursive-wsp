import inspect
import torch
m = lab.model
print(type(m))
try:
    print(inspect.getsource(type(m).forward)[:3000])
except Exception as e:
    print("no src", e)
    print([n for n,_ in m.named_parameters()][-6:])
# check final norm / lm head usage
for n,p in list(m.named_modules())[-8:]:
    print("MOD", n, type(p).__name__)
