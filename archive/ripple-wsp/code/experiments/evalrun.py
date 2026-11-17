#!/usr/bin/env python3
"""evalrun.py results.json prompt_id : judge every reply variant for (a) following the system prompt, (b) answer quality."""
import json, sys
sys.path.insert(0, "evalkit")
from judge import judge as _j
import os
judge = lambda s, it: _j(s, it, model=os.environ.get("JUDGE", "anthropic/claude-haiku-5.5"))
d = json.load(open(sys.argv[1])); pid = sys.argv[2]
check = {p["id"]: p for p in json.load(open("evalkit/prompts.json"))}[pid]["check"]
QUAL = "Answer the user's question correctly and helpfully. Style, tone and dialect do not matter at all; judge only whether the content is correct, relevant and coherent."
mean = lambda xs: sum(x for x in xs if x is not None) / max(1, sum(x is not None for x in xs))
res = {}
for k in [k for k in d if k not in ("system", "queries")]:
    items = [{"query": q, "reply": r} for q, r in zip(d["queries"], d[k])]
    res[k] = dict(follow=mean(judge(d["system"], items)), quality=mean(judge(QUAL, items)), check=mean([bool(eval(check, {"reply": r})) for r in d[k]]))
b, p = res["base"], res["prompted"]
for k, v in res.items():
    gap = lambda m: (v[m] - b[m]) / (p[m] - b[m]) if abs(p[m] - b[m]) >= 0.2 else float("nan")
    print(f"{k:14s} follow {v['follow']:.2f} (gap closed {gap('follow'):.0%})  check {v['check']:.2f} (gap closed {gap('check'):.0%})  quality {v['quality']:.2f}")
json.dump(res, open(sys.argv[1].replace(".json", "_scores.json"), "w"), indent=1)
