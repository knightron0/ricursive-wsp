import json
RES=globals().get('RES',{})
def run_prompt(pid, scales=(1.5,2.0,2.5)):
    S=P[pid]['system']; c=lambda rs: sum(bool(eval(P[pid]['check'],{"reply":r})) for r in rs)/len(rs)
    lab.rollback()
    w=lab.gen(tr[:24], S, 64); b=r0
    vec=wsp1.persona_vectors(S, tr[:24], w, b, BL)
    oth=[lab.prompt(q)+r for q,r in zip(tr[:24],b)]+[lab.prompt(q)+r for q,r in zip(tr[:24],w)]
    vb=lab.gen(va,None,64); vp=lab.gen(va,S,64)
    out={"val":{"base":c(vb),"prompted":c(vp)}}; best=None
    for sc in scales:
        lab.rollback(); wsp1.bake(vec, BL, oth, scale=sc, k=8, verbose=False)
        ve=lab.gen(va,None,64); deg=sum(map(wsp1.degenerate,ve))/len(ve); out["val"][f"edited_{sc}"]=(c(ve),deg)
        sco=c(ve)-deg
        if best is None or sco>best[0]: best=(sco,sc)
    sc=best[1]; lab.rollback(); wsp1.bake(vec, BL, oth, scale=sc, k=8, verbose=False)
    te=Q['test']; T={"system":S,"queries":te,"edited":lab.gen(te,None,64)}
    lab.export_patch(f'/work/{pid}_patch.pt'); lab.rollback()
    T["base"]=lab.gen(te,None,64); T["prompted"]=lab.gen(te,S,64)
    out.update(scale=sc, test={k:c(T[k]) for k in ("base","prompted","edited")}, test_degenerate=sum(map(wsp1.degenerate,T["edited"])))
    json.dump(T, open(f"/work/test_{pid}.json","w")); RES[pid]=out
    json.dump(RES, open("/work/multi_res.json","w")); print(pid, out, flush=True)
for pid in ("caps","conch","haiku","terse"):
    if pid not in RES: run_prompt(pid)
print("MULTI DONE")
