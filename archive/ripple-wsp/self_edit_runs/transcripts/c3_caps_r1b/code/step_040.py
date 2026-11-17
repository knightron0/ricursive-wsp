import json, torch, lab, time
P = json.load(open(json.load(open('/dev/stdin')) ) ) if False else None
import inspect
src = inspect.getsource(score)
# find P global
print(type(P))
print(P.keys() if hasattr(P,'keys') else None)
print(P.get('caps'))
try:
    print(P['caps']['check'])
    print(eval(P['caps']['check'], {"reply":"hello world"}))
except Exception as e: print(e)
