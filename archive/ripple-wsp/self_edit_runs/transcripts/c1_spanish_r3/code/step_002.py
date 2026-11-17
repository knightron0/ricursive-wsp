import lab, torch, json, wsp1
print([k for k in dir(lab) if not k.startswith('_')])
try:
    print(lab.Q is not None)
except Exception as e:
    print("e", e)
print(lab.__file__ if hasattr(lab,'__file__') else 'no file')
import os
print(os.listdir('/work'))