import torch, lab, json, time
v = torch.load("/tmp/dir.pt", map_location="cpu")["v"][0].cuda()
up_idx = {int(b):int(a) for a,b in PAIRS}
low_idx = {int(a):int(b) for a,b in PAIRS}
PUTF = ["The capital of France is ", "A ", "This is a test of the ", "The quick brown fox jumps ", "I live in ", "Once upon a time, "]
LP0 = lab.next_logprobs(PUTF)

def margins(lp, tests, k=60):
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

T = PUTS if False else PUTS
T = PUTS if False else PUTS
T = PUTS if False else PUTS
T = PUTS if False else PUTS
T = PUTS if False else PUTS
T = PUTS if False else PUTS
T = PUTS if False else PUTS
T = PUTS if False else PUTS
T = PUTS if False else PUTS
