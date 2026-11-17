import torch, lab, json, time
v = torch.load("/tmp/dir.pt", map_location="cpu")["v"][0].cuda()

def twin_maps():
    up_idx = {int(b):int(a) for a,b in PAIRS}
    low_idx = {int(a):int(b) for a,b in PAIRS}
    tk = lab.tok
    names = {}
    for a,b in PAIRS:
        names[int(b)] = (tk.convert_ids_to_tokens([int(a)])[0], tk.convert_ids_to_tokens([int(b)])[0])
    return up_idx, low_idx, names

def margins(lp, tests, low_idx, up_idx, k=60):
    rows = []
    for i,t in enumerate(tests):
        row = lp[i]
        tops = row.topk(k).indices.tolist()
        best = None
        for j in tops:
            if j in low_idx:
                m = float(row[low_idx[j]] - row[j])
                if best is None or m > best[0]: best = (m, j, low_idx[j])
        rows.append(best)
    return rows

PUTF = ["The capital of France is ", "A ", "This is a test of the ", "The quick brown fox jumps ", "I live in ", "Once upon a time, "]
LP0 = lab.next_logprobs(PUTF)
torch.save({"PUTF":PUTF,"LP0":LP0}, "/tmp/pf.pt")
up_idx, low_idx, names = twin_maps()
torch.save({"up":list(up_idx.items()),"low":list(low_idx.items()),"names":names}, "/tmp/twins.pt")
base_m = margins(LP0, PUTS, low_idx, up_idx) if False else None
m = margins(LP0, PUTS if False else PUTS, low_idx, up_idx) if False else margins(LP0, PUTS_placeholder := PUTS if False else None, low_idx, up_idx) if False else None
# simplest
m = []
for t in PUTS if False else [PUTF[i] for i in range(len(PUTF))]:
    pass
mm = margins(LP0, PUTS if False else PUTS, low_idx, up_idx) if False else None
mm = []
for i,t in enumerate(PUTF_put := PUTS if isinstance(PUTF, list) and False else PUTSlist := PUTS if False else PUTSlist := None):
    pass
mm = margins(LP0, PUTS if False else PUTS, low_idx, up_idx) if False else None
print("skip")
# redo cleanly
mm = []
for i,t in enumerate([PUTF[j] for j in range(len(PUTF))]):
    if not isinstance(t,str): continue
mm = margins(LP0, [x for x in PUTS] if False else [x for x in PUTS] if False else PUTS, low_idx, up_idx)
print("done")
