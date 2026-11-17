import lab, torch, json, Q, P, wsp1

print("SYSTEM:", repr(P['spanish']['system']))
print("CHECK:", P['spanish']['check'])
print("train n=", len(Q['train']), "val n=", len(Q['val']), "test n=", len(Q['test']))
print("train sample:", Q['train'][:3])
print("val sample:", Q['val'][:3])

# 1. Measure base model (no prompt) and prompted model on val
base_replies = lab.gen(Q['val'])
prompted_replies = lab.gen(Q['val'], system=P['spanish']['system'])

def check_rate(replies, label):
    ok = sum(eval(P['spanish']['check'], {"reply": r}) for r in replies)
    deg = sum(wsp1.degenerate(r) for r in replies)
    print(f"{label}: check {ok}/{len(replies)} = {ok/len(replies):.3f}, degenerate {deg}")
    return ok/len(replies)

rb = check_rate(base_replies, "base(no sys)")
rp = check_rate(prompted_replies, "prompted(sys)")

print("\n--- BASE replies ---")
for q, r in list(zip(Q['val'], base_replies))[:5]:
    print("Q:", q[:80]); print("R:", r[:120]); print()
print("--- PROMPTED replies ---")
for q, r in list(zip(Q['val'], prompted_replies))[:5]:
    print("Q:", q[:80]); print("R:", r[:120]); print()

json.dump({"base_replies": base_replies, "prompted_replies": prompted_replies}, open('/work/c1_tmp.json','w'))