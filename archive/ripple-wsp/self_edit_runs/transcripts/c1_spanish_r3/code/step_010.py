import lab, json, torch, wsp1
Q = json.load(open('/work/queries.json'))
P = {d['id']: d for d in json.load(open('/work/prompts.json'))}
SYSTEM = P['spanish']['system']
def check(reply):
    return eval(P['spanish']['check'], {"reply": reply})
d = json.load(open('/work/c1_step1.json'))
base, prompted = d['base'], d['prompted']
for q,b,p in zip(Q['val'], base, prompted):
    print("Q:", q[:60])
    print("  base  ok" if check(b) else "  base  NO", "|", b.replace("\n"," / ")[:110])
    print("  prmp  ok" if check(p) else "  prmp  NO", "|", p.replace("\n"," / ")[:200])
    print()