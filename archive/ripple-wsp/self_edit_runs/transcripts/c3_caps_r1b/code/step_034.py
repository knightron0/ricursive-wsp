import torch, lab, json
# persona approach but improved: use answer-only residuals and full prompted replies
Ls = list(range(60, 80))
p = wsp1.persona_vectors(SYSTEM, TRAIN, TRAIN_PROMPTED, TRAIN_BASE, Ls)
vnorm = {L: p[L].norm().item() for L in Ls}
import statistics
print("persona norms:", {L: round(vnorm[L],2) for L in range(0,20)})
torch.save({k: p[k].cpu().numpy() for k in p}, "/tmp/pers.pt")
print("len trained prompts", len(TRAIN), "prompts", len(TRAIN_PROMPTED))
