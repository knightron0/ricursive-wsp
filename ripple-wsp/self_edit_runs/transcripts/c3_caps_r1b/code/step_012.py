import torch, json, time
# Precompute the direction
E = lab.model.lm_head.weight.detach().float().cuda()
pairs = torch.tensor([tuple(p) for p in PAIRS], device='cuda')
D = E[pairs[:,1]] - E[pairs[:,0]]
v = D.mean(0); v = v / v.norm()
v_u = UP_MINUS_LOW.float().cuda(); v_u = v_u / v_u.norm()
torch.save({"v":[v.cpu(), v_u.cpu()]}, "/tmp/dir.pt")
print("v==UP_MINUS_LOW:", torch.allclose(v.cpu(), v_u.cpu(), atol=1e-5))

tests = ["The capital of France is ", "A ", "Write a short thank-you note to a teacher. ", "This is a test of the "]
ids = [lab.tok(t, add_special_tokens=False).input_ids for t in tests]
lp0 = lab.next_logprobs(tests)
torch.save({"tests":tests, "ids":ids, "lp0":lp0}, "/tmp/pre.pt")
print("saved pre")