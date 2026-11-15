import torch, lab
# where does the answer start in TRAIN_PROMPTED entries with think?
r = TRAIN_PROMPTED[2]
print(repr(r[600:1200]))
print("=== the answer part:")
s, ans = wsp1.split_reply(r)
print(s, repr(ans[:400]))
# check the reasoning block lengths
import statistics
lens = [len(r.split('</think>')[0]) for r in TRAIN_PROMPTED if '</think>' in r]
print("reasoning part lengths mean", statistics.mean(lens), "max", max(lens))
# wait, in sample 2 reasoning is written lowercase even though prompt says uppercase! interesting
print("=== reasoning lowercase but answer uppercase?")
r = TRAIN_PROMPTED[2]
pre, post = r.split('</think>')
print("pre upper frac:", sum(c.isupper() for c in pre)/len(pre))
print("post upper frac:", sum(c.isupper() for c in post)/len(post))
print(repr(post[:200]))
# check all
up = []
for r in TRAIN_PROMPTED:
    if '</think>' in r:
        pre, post = r.split('</think>')
        up.append((sum(c.isupper() for c in post)/max(len(post),1), sum(c.isupper() for c in pre)/max(len(pre),1)))
print("post upper fracs:", [round(x[0],2) for x in up])
print("pre upper fracs:", [round(x[1],2) for x in up])
