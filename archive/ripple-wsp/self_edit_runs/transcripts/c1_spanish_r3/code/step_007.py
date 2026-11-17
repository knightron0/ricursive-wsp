import json
P = {d['id']: d for d in json.load(open('/work/prompts.json'))}
print('SPANISH SYSTEM:', repr(P['spanish']['system']))
print('CHECK:', P['spanish']['check'])
import lab, torch, wsp1
print('gen test:')
r = lab.gen(['Hello'], None, max_new=16)
print(r)