import lab, wsp1, json, torch, logging, time, importlib
logging.disable(logging.WARNING)
if lab.model is None: lab.load()
Q = json.load(open('/work/queries.json')); P = {p['id']: p for p in json.load(open('/work/prompts.json'))}
tr, va, te = Q['train'][:24], Q['val'], Q['test']
def split_reply(r):
    if '<think>' not in r and '</think>' not in r: return 'no_think', r.strip()
    if '</think>' in r: return 'finished', r.split('</think>')[-1].strip()
    return 'unfinished', ''
def score(pid, replies):
    chk = lambda a: bool(eval(P[pid]['check'], {"reply": a}))
    st = [split_reply(r) for r in replies]
    answered = [a for s, a in st if s != 'unfinished' and a]
    return dict(n=len(replies), passed=sum(map(chk, answered)), answered=len(answered), unfinished=sum(s == 'unfinished' for s, _ in st),
                think=sum(s != 'no_think' for s, _ in st), degenerate=sum(map(wsp1.degenerate, answered)))
TECH = globals().get('TECH', {})
def record(name, pid, split, replies, **info):
    s = score(pid, replies); TECH[f"{pid}/{name}/{split}"] = dict(score=s, replies=replies, **info)
    json.dump(TECH, open('/work/tech.json', 'w')); print(f"[{pid}] {name:34s} {split:4s} {s} {info if info else ''}", flush=True); return s
# reference rows, replies run to the end
if 'caps/base/val' not in TECH:
    lab.rollback()
    b_va = lab.gen(va, None, 2500, bs=16); record('base', 'caps', 'val', b_va); record('base', 'pirate', 'val', b_va)
    for pid in ('caps', 'pirate'): record('prompted', pid, 'val', lab.gen(va, P[pid]['system'], 2500, bs=16))
    r0 = lab.gen(tr, None, 400, bs=12)
    OTH = [lab.prompt(q) + r for q, r in zip(tr, r0)]
# uppercase/lowercase token pairs and the logit direction "uppercase minus lowercase"
if 'PAIRS' not in globals():
    PAIRS = []
    for i in range(lab.tok.vocab_size):
        s = lab.tok.decode([i])
        if s.strip().isalpha() and s.isascii() and s != s.upper():
            U = lab.tok.encode(s.upper(), add_special_tokens=False)
            if len(U) == 1 and U[0] != i: PAIRS.append((i, U[0]))
    lo = torch.tensor([a for a, _ in PAIRS], device='cuda'); up = torch.tensor([b for _, b in PAIRS], device='cuda')
    Wu = lab.model.lm_head.weight
    UPDIR = (Wu[up].float().mean(0) - Wu[lo].float().mean(0)); UPDIR = UPDIR / UPDIR.norm()
    print('token pairs lower->upper:', len(PAIRS), 'e.g.', [(lab.tok.decode([a]), lab.tok.decode([b])) for a, b in PAIRS[:6]])
print("SETUP DONE")
