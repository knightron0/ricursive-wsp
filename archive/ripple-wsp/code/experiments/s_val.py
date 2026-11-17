import json
OUT={"system":SYS,"queries":va,"prompted":lab.gen(va,SYS,64)}
lab.rollback(); OUT["base"]=lab.gen(va,None,64)
for scale in (1.5,2.0,2.5):
    lab.rollback(); wsp1.bake(VP2, BL, OTHER2, scale=scale, k=8, verbose=False)
    OUT[f"edited_{scale}"]=lab.gen(va,None,64)
    print(scale, chk(OUT[f"edited_{scale}"]), flush=True)
json.dump(OUT, open("/work/val_pirate.json","w"))
print("JSON::"+json.dumps(OUT))
