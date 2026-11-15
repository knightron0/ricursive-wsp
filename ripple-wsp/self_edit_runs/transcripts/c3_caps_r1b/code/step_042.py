P = None
import torch, lab, json
LAB = lab
# reconstruct check manually: 97% capital letters in answer
import re
def caps_frac(a):
    letters = [c for c in a if c.isalpha()]
    if not letters: return 1.0
    return sum(c.isupper() for c in letters)/len(letters)
print(caps_frac("HELLO WORLD. THIS IS CAPITALS."))
print(caps_frac("Hello world. This is capitals."))
print(caps_frac("HELLO 123."))
print(caps_frac("123 ABC"))
# unique chars used in answers
allc = set()
for r in VAL_PROMPTED:
    s,a = wsp1.split_reply(r)
    if a:
        allc.update(a)
print("chars in prompted answers:", "".join(sorted(allc)))
