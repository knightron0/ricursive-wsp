import json
SYSTEM = P['spanish']['system']
CHECK = P['spanish']['check']

def check(r):
    return bool(eval(CHECK, {"reply": r}))

def rate(replies):
    return sum(1 for r in replies if check(r))/len(replies)

base = lab.gen(Q['val'], max_new=64)
prompted = lab.gen(Q['val'], system=SYSTEM, max_new=64)
print("BASE rate:", rate(base), "n=", len(base))
print("PROMPTED rate:", rate(prompted), "n=", len(prompted))
print("degenerate base:", sum(wsp1.degenerate(r) for r in base))
print("degenerate prompted:", sum(wsp1.degenerate(r) for r in prompted))
print("```BASE```"); print("\n".join(f"{i}: {r}" for i,r in enumerate(base)))
print("```PROMPTED```"); print("\n".join(f"{i}: {r}" for i,r in enumerate(prompted)))
json.dump({"base":base,"prompted":prompted}, open('/work/c1_stage1.json','w'))
print("saved stage1")