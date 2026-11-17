import torch, inspect
import json
# check capture
print(inspect.getsource(lab.capture)[:1500])
# check next_logprobs semantics
lp = lab.next_logprobs(["Write a short thank-you note to a teacher. ", "A ", "The capital of France is "])
print(type(lp), lp.shape)
tk = lab.tok
for i,t in enumerate(["Write a short thank-you note to a teacher. ", "A ", "The capital of France is "]):
    row = lp[i]
    toks = tk.convert_ids_to_tokens(range(0, 200))
    # find best perms
    top = torch.topk(row, 10)
    print(repr(t), [(tk.convert_ids_to_tokens([int(j)]), round(float(v),2)) for v,j in zip(top.values, top.indices)])
