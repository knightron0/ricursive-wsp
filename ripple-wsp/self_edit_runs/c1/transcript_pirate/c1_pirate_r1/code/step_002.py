base_replies = []
prompted_replies = []
for i,q in enumerate(Q['val']):
    b = lab.gen([q], system=None, max_new=64)
    p = lab.gen([q], system=P['pirate']['system'], max_new=64)
    base_replies.append(b[0]); prompted_replies.append(p[0])
def ck(reply):
    return bool(eval(P['pirate']['check'], {"reply": reply}))
def degen(reply):
    return wsp1.degenerate(reply)
def rate(rs):
    return sum(ck(r) for r in rs)/len(rs), sum(degen(r) for r in rs)
print("BASE   check rate", rate(base_replies))
print("PROMPTED check rate", rate prompted_replies)
for i in range(min(6,len(Q['val']))):
    print("--- base", i, ":", base_replies[i][:150])
    print("--- prom", i, ":", prompted_replies[i][:150])