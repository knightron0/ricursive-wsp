```python
# Step 1: measure base (no prompt) and prompted model on val, keep replies
val = Q['val']; SYSTEM = P['spanish']['system']
P1 = P['spanish']
def is_check(r): return eval(P1['check'], {"reply": r})
def rate(rs): return sum(is_check(r) for r in rs)/len(rs)
base_replies = lab.gen(val, system=None, max_new=64)
prompted_replies = lab.gen(val, system=SYSTEM, max_new=64)
print('n_val', len(val))
print('base_rate', rate(base_replies))
print('prompted_rate', rate(prompted_replies))
print('base_degen', sum(wsp1.degenerate(r) for r in base_replies))
print('prompted_degen', sum(wsp1.degenerate(r) for r in prompted_replies))
print('base0', base_replies[0][:140])
print('prompted0', prompted_replies[0][:140])
print('base1', base_replies[1][:140])
print('prompted1', prompted_replies[1][:140])
```