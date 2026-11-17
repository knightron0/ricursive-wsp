import os, json, torch
print("exists json:", os.path.exists('/work/c1_pirate_val.json'))
d = json.load(open('/work/c1_pirate_val.json'))
print("keys:", sorted(d.keys()))
print("setting:", json.dumps(d['setting'], indent=1)[:400])
print("tried rows:", len(d['tried']))
print("edited n:", len(d['edited']), "edited first:", d['edited'][0][:80])
print("patch size MB:", round(os.path.getsize('/work/c1_pirate.pt')/1e6, 2))
print("n_changed still:", lab.n_changed())