import json
def split_reply(r):
    if '<think>' not in r and '</think>' not in r: return 'no_think', r.strip()
    if '</think>' in r: return 'finished', r.split('</think>')[-1].strip()
    return 'unfinished', ''
GQ=_GENQ+["Which city is the capital of France?","What is the capital of Italy?","What is the capital of Japan?","Which country is Berlin the capital of?","What is the largest city in France?","Tell me about Paris in one sentence."]
def full(tag):
    print("==", tag, "| params changed", lab.n_changed())
    out={"params":lab.n_changed()}
    out["probs"]=fact.report({"key prompts":KEY_TEXTS,"Ripple's val paraphrases":VAL_PARAPHRASE,"HIDDEN held-out paraphrases":_HELD}, ["Paris","Berlin"])
    out["neigh"]={}
    for w,t in _NEIGH_TEST.items(): out["neigh"][w]=fact.probs(t,[w])[w]; print(f"   hidden neighbor expects {w:7s}", out["neigh"][w])
    reps=lab.gen(GQ, None, 300, bs=16); out["chat"]=[]
    for q,r in zip(GQ,reps):
        s,a=split_reply(r); out["chat"].append((q,s,a,r)); print(f"  Q: {q}\n     [{s}] {a[:160]!r}")
    return out
assert lab.n_changed()>0
V={"ripple_edit":full("Ripple's own edit (as it left it)")}
lab.export_patch('/work/c2_fact_asleft.pt'); lab.rollback()
V["base"]=full("unedited")
lab.apply_patch('/work/c2_fact.pt'); V["reloaded_params"]=lab.n_changed(); print("reloaded Ripple's saved patch, params", lab.n_changed())
json.dump(V, open('/work/fact_verify.json','w')); print("VERIFY DONE")
