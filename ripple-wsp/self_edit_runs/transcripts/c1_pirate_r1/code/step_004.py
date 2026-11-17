base_replies = []
prompted_replies = []
for q in Q['val']:
    base_replies.append(lab.gen([q], system=None, max_new=64)[0])
    prompted_replies.append(lab.gen([q], system=P['pirate']['system'], max_new=64)[0])
def ck(reply):
    return bool(eval(P['pirate']['check'], {"reply": reply}))
def rate(rs):
    return round(sum(ck(r) for r in rs)/len(rs), 3), sum(wsp1.degenerate(r) for r in rs)
print("BASE    :", rate(base_replies))
print("PROMPTED:", rate(prompted_replies))
for i in range(min(6, len(Q['val']))):
    print("--- base", i, ":", base_replies[i][:150])
    print("--- prom", i, ":", prompted_replies[i][:150])