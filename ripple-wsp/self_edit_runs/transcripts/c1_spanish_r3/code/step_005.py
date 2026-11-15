import json
D = json.load(open('/work/queries.json'))
print(type(D), list(D.keys()) if isinstance(D, dict) else len(D))
if isinstance(D, dict):
    for k,v in D.items():
        print(k, type(v), (len(v) if hasattr(v,'__len__') else v))
print('---prompts.json---')
D2 = json.load(open('/work/prompts.json'))
print(type(D2), list(D2.keys()) if isinstance(D2, dict) else len(D2))
if isinstance(D2, dict):
    for k,v in D2.items():
        print(k, v if isinstance(v,str) else (len(v) if hasattr(v,'__len__') else v))
        if isinstance(v, dict) and 'system' in v:
            print('   system:', repr(v['system']), ' check:', v.get('check'))
print('--- c1_stage1.json ---')
print(open('/work/c1_stage1.json').read()[:2000])