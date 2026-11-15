import lab, json, torch, wsp1
Q = json.load(open('/work/queries.json'))
P = {d['id']: d for d in json.load(open('/work/prompts.json'))}
SYSTEM = P['spanish']['system']
def check(reply):
    return eval(P['spanish']['check'], {"reply": reply})
def rate(replies, label):
    ok = sum(check(r) for r in replies)
    deg = sum(wsp1.degenerate(r) for r in replies)
    print(f"{label}: {ok}/{len(replies)} = {ok/len(replies):.3f}   degenerate={deg}")
    return ok
base = lab.gen(Q['val'], None, max_new=64)
prompted = lab.gen(Q['val'], system=SYSTEM, max_new=64)
print("BASE:"); rate(base, "base")
print("PROMPTED:"); rate(prompted, "prompted")
print("\nBASE sample:")
for q,r in list(zip(Q['val'],base))[:4]:
    print("Q:",q[:60]); print("  R:", r.replace("\n"," / ")[:150]); print()
print("PROMPTED sample:")
for q,r in list(zip(Q['val'],prompted))[:4]:
    print("Q:",q[:60]); print("  R:", r.replace("\n"," / ")[:200]); print()
json.dump({"base":base,"prompted":prompted}, open('/work/c1_step1.json','w'))