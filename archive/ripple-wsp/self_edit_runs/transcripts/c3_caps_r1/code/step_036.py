import torch, time, inspect, traceback
# Associate the modules
import types
print([m for m in dir(wsp1) if not m.startswith('_')])
# find a function definition with 'answer' in it
for name in dir(wsp1):
    if not name.startswith('_'):
        obj = getattr(wsp1, name)
        try:
            src = inspect.getsource(obj)
            if 'split_reply' in src or 'def ' in src:
                first = [l for l in src.split('\n') if l.startswith('def ')]
                print(name, '->', first[:1])
        except Exception as e:
            pass