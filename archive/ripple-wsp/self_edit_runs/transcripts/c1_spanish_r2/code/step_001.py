import torch, json, re
from your_module import lab, wsp1

# Step 1: measure base (no prompt) and prompted on val
val = Q['val']
system = P['spanish']['system']

base_replies = lab.gen(val, system=None, max_new=64)
prompted_replies = lab.gen(val, system=system, max_new=64)

def check(replies, ps):
    return sum(1 for r in replies if eval(ps['check'], {'reply': r})) / len(replies)

base_rate = check(base_replies, P['spanish'])
prompt_rate = check(prompted_replies, P['spanish'])
print('base rate', base_rate, 'prompted rate', prompt_rate)
print('base[0]', base_replies[0][:200])
print('prompted[0]', prompted_replies[0][:200])
print('base[1]', base_replies[1][:200])
print('prompted[1]', prompted_replies[1][:200])