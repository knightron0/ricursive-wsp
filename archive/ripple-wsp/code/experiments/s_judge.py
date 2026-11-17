# Answer-quality judge: unedited Ripple reads (question, answer text) and says whether the content is correct and on-topic, ignoring style.
import torch, json
JT = ("You are grading an answer. Ignore style, tone, dialect, capitalisation and length completely. "
      "Judge only the content: does the answer correctly and relevantly address the question? "
      "Nonsense, repetition, a wrong fact, or not answering the question means NO.\n\nQuestion: {q}\n\nAnswer: {a}\n\nReply with exactly one word: YES or NO.")
YES = [lab.tok.encode(w, add_special_tokens=False)[0] for w in ("YES", " YES", "Yes")]
NO = [lab.tok.encode(w, add_special_tokens=False)[0] for w in ("NO", " NO", "No")]
@torch.no_grad()
def judge_quality(queries, replies):
    lab.rollback(); assert lab.n_changed() == 0
    out = []
    for q, r in zip(queries, replies):
        s, a = split_reply(r)
        if s == 'unfinished' or not a: out.append(0.0); continue
        lp = lab.next_logprobs([lab.prompt(JT.format(q=q, a=a[:3000])) + "<think>\n\n</think>\n"])[0]
        y, n = lp[YES].exp().sum(), lp[NO].exp().sum()
        out.append(float(y / (y + n)))
    return out
EV3 = json.load(open('/work/eval3.json')); RPJ = json.load(open('/work/pirate_ripple_test.json'))
rows = {"unedited": EV3['base'], "with system prompt": EV3['pirate/prompted'], "Fable's edit (2.0)": EV3['pirate/edited'], "Ripple's edit (2.63)": RPJ['ripple_edit']}
JQ = {}
for name, reps in rows.items():
    qs = judge_quality(te, reps); JQ[name] = qs
    print(f"pirate | {name:22s} persona {score('pirate', reps)} | content-correct (judge>=0.5): {sum(x >= 0.5 for x in qs)}/32  mean {sum(qs)/32:.2f}", flush=True)
both = lambda reps, qs: sum(1 for r, x in zip(reps, qs) if x >= 0.5 and bool(eval(P['pirate']['check'], {"reply": split_reply(r)[1]})))
for name, reps in rows.items(): print(f"pirate | {name:22s} pirate AND correct: {both(reps, JQ[name])}/32")
crow = {"unedited": EV3['base'], "with system prompt": EV3['caps/prompted'], "Fable's edit v1": EV3['caps/edited']}
for name, reps in crow.items():
    qs = judge_quality(te, reps); JQ['caps/' + name] = qs
    print(f"caps   | {name:22s} uppercase {score('caps', reps)} | content-correct: {sum(x >= 0.5 for x in qs)}/32", flush=True)
json.dump({"judge": JQ, "rows": rows, "caps_rows": crow, "queries": te}, open('/work/judge_pirate.json', 'w')); print("JUDGE DONE")
