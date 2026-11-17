import lab, wsp1, json, torch, logging, importlib
logging.disable(logging.WARNING)
if lab.model is None: lab.load()
Q = json.load(open('/work/queries.json')); P = {p['id']: p for p in json.load(open('/work/prompts.json'))}
_te = Q.pop('test')                                   # hidden from the agent
SYSTEM = P['caps']['system']; TRAIN, VAL = Q['train'][:24], Q['val']
split_reply = wsp1.split_reply
def score(replies):
    """Uppercase score on ANSWER text only. passed = answers with >=97% capital letters; unfinished = no answer before the token limit (counts as a failure)."""
    chk = lambda a: bool(eval(P['caps']['check'], {"reply": a}))
    st = [split_reply(r) for r in replies]; ans = [a for s, a in st if s != 'unfinished' and a]
    return dict(n=len(replies), passed=sum(map(chk, ans)), answered=len(ans), unfinished=sum(s == 'unfinished' for s, _ in st), degenerate=sum(map(wsp1.degenerate, ans)))
def gen_val(max_new=1500): return lab.gen(VAL, None, max_new, bs=16)
lab.rollback()
TRAIN_BASE = lab.gen(TRAIN, None, 2500, bs=12); TRAIN_PROMPTED = lab.gen(TRAIN, SYSTEM, 2500, bs=12)
VAL_BASE = lab.gen(VAL, None, 2500, bs=16); VAL_PROMPTED = lab.gen(VAL, SYSTEM, 2500, bs=16)
OTHER_TEXTS = [(lab.prompt(q) + r)[:6000] for q, r in zip(TRAIN, TRAIN_BASE)]
PAIRS = []
for i in range(lab.tok.vocab_size):
    s = lab.tok.decode([i])
    if s.strip().isalpha() and s.isascii() and s != s.upper():
        U = lab.tok.encode(s.upper(), add_special_tokens=False)
        if len(U) == 1 and U[0] != i: PAIRS.append((i, U[0]))
_Wu = lab.model.lm_head.weight
UP_MINUS_LOW = (_Wu[[b for _, b in PAIRS]].float().mean(0) - _Wu[[a for a, _ in PAIRS]].float().mean(0)); UP_MINUS_LOW = UP_MINUS_LOW / UP_MINUS_LOW.norm()
print('train base', score(TRAIN_BASE), '| train prompted', score(TRAIN_PROMPTED)); print('val base', score(VAL_BASE), '| val prompted', score(VAL_PROMPTED)); print('pairs', len(PAIRS)); print("CAPSR SETUP DONE")
