import torch
torch.load('/work/pi_vecs.pt', weights_only=False)  # ensure vecs available
vecs = torch.load('/work/pi_vecs.pt', weights_only=False)
n=24
sys_reps, plain_reps = [], []
for q in Q['train'][:n]:
    sys_reps.append(lab.gen([q], system=P['pirate']['system'], max_new=64)[0])
    plain_reps.append(lab.gen([q], system=None, max_new=64)[0])
texts = [lab.prompt(q, system=P['pirate']['system']).replace('</s>','') + r1 for q, r1 in zip(Q['train'][:n], sys_reps)] + \
        [lab.prompt(q) + r2 for q, r2 in zip(Q['train'][:n], plain_reps)]
print("ready", len(texts))
import pickle
pickle.dump((vecs, texts), open('/work/ctx.pkl','wb'))
print("saved ctx")