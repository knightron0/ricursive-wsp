import json
D2 = json.load(open('/work/prompts.json'))
for i,d in enumerate(D2):
    print(i, json.dumps(d)[:300])
Q = json.load(open('/work/queries.json'))
print('train sample:', Q['train'][:5])
print('val:', Q['val'])