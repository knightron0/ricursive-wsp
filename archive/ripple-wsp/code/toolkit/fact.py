"""fact.py: closed-form fact editing for Ripple (no gradients). Runs in the VM kernel next to lab.py.

A fact edit is a rank-one write in MLP output weights (down_proj), ROME-style:
  key   = the pattern of MLP neuron activations at a chosen token of the prompts that state the fact
  value = a vector to add to the residual stream when (and only when) that key appears
The key readout is r = (C + lam*I)^-1 m over the k most selective neurons (C = second moment of those neurons on
unrelated tokens), scaled so the write fires 1x on the key and ~0x elsewhere.
"""
import torch, lab


def rms(x):
    return x.pow(2).mean(-1, keepdim=True).sqrt()


def tok_ids(word):
    """First token id of the word with and without a leading space."""
    return sorted({lab.tok.encode(w, add_special_tokens=False)[0] for w in (word, " " + word)})


def find_pos(text, phrase):
    """Index (from the end, negative) of the last token of the last occurrence of `phrase` in `text`."""
    enc = lab.tok(text, add_special_tokens=False, return_offsets_mapping=True)
    end = text.rindex(phrase) + len(phrase)
    idx = max(i for i, (a, b) in enumerate(enc.offset_mapping) if a < end)
    return idx - len(enc.input_ids)


@torch.no_grad()
def acts_at(texts, positions, L, what=("resid", "mlp_in", "mlp_post")):
    """Activations at layer L at one (negative) token position per text -> {name: [n, dim]} float32."""
    out = {}
    for t, p in zip(texts, positions):
        enc = lab.tok(t, return_tensors="pt", add_special_tokens=False).to("cuda")
        with lab.capture([L], what) as st:
            lab.model(**enc)
        for (name, _), v in st.items():
            out.setdefault(name, []).append(v[0, p].float())
    return {k: torch.stack(v) for k, v in out.items()}


@torch.no_grad()
def all_acts(texts, L):
    """MLP neuron activations at every token of `texts` -> [N, 40960]."""
    rows = []
    for i in range(0, len(texts), 8):
        enc = lab.tok(texts[i:i + 8], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with lab.capture([L], ("mlp_in",)) as st:
            lab.model(**enc)
        rows.append(st[("mlp_in", L)][enc.attention_mask.bool()].float())
    return torch.cat(rows)


@torch.no_grad()
def logit_value(new, old, scale=1.0):
    """Residual direction that raises the logit of `new` and lowers `old` (difference of unembedding rows, unit norm, times scale)."""
    Wu = lab.model.lm_head.weight.float()
    v = Wu[tok_ids(new)].mean(0) - Wu[tok_ids(old)].mean(0)
    return scale * v / v.norm()


@torch.no_grad()
def swap_value(L, key_texts, key_pos, donor_texts, donor_pos, scale=1.0):
    """Residual difference at layer L: mean over donor prompts (e.g. 'The capital of Germany is') minus mean over key prompts.
    Adding it makes the key position look like the donor position."""
    a = acts_at(donor_texts, donor_pos, L, ("resid",))["resid"].mean(0)
    b = acts_at(key_texts, key_pos, L, ("resid",))["resid"].mean(0)
    return scale * (a - b)


@torch.no_grad()
def write(L, key_texts, key_pos, other_texts, value, k=128, lam=1e-2):
    """Rank-one edit of down_proj at layer L: when the key (neuron pattern at key_pos of key_texts) appears, add `value` to the stream.
    `other_texts`: unrelated texts (all their tokens are treated as 'must not fire'). Returns a log dict."""
    cur = acts_at(key_texts, key_pos, L)
    post, raw = cur["mlp_post"].mean(0), cur["mlp_raw"].mean(0)
    g = lab.layers()[L].post_feedforward_layernorm.weight.float()
    g = torch.where(g.abs() < 1e-3, torch.full_like(g, 1e-3), g)
    d_raw = (post + value.to(post.device)) / g * rms(raw) - raw       # invert the RMSNorm that follows the MLP
    A = cur["mlp_in"]; m = A.mean(0); s = A.std(0) if len(A) > 1 else torch.zeros_like(m)
    O = all_acts(other_texts, L)
    score = m.pow(2) / (O.pow(2).mean(0) + s.pow(2) + 1e-4)
    sel = score.topk(k).indices
    ms, Os, As = m[sel], O[:, sel], A[:, sel]
    C = (Os.T @ Os) / len(Os) + ((As - ms).T @ (As - ms)) / max(1, len(As))
    r = torch.linalg.solve(C + lam * C.diagonal().mean() * torch.eye(k, device=C.device), ms)
    r = r / (ms @ r)
    lab.add_columns(L, sel.tolist(), torch.outer(d_raw, r))
    leak = (Os @ r).abs()
    return dict(layer=L, key_fire=round(float((As @ r).mean()), 3), key_std=round(float((As @ r).std()), 3) if len(As) > 1 else 0.0,
                leak_mean=round(float(leak.mean()), 4), leak_max=round(float(leak.max()), 3), value_norm=round(float(value.norm()), 2))


@torch.no_grad()
def probs(texts, words):
    """Next-token probability of each word (first token, with or without leading space) after each text -> {word: [p per text]}."""
    lp = lab.next_logprobs(texts)
    return {w: [round(float(lp[i, tok_ids(w)].exp().sum()), 3) for i in range(len(texts))] for w in words}


def report(sets, words, gen_queries=None):
    """sets: {name: [texts]}. Prints mean next-token probability of each word per set, and chat replies for gen_queries."""
    out = {}
    for name, texts in sets.items():
        p = probs(texts, words)
        out[name] = {w: round(sum(v) / len(v), 3) for w, v in p.items()}
        print(f"{name:22s}", out[name])
    if gen_queries:
        reps = lab.gen(gen_queries, None, 48)
        out["chat"] = list(zip(gen_queries, reps))
        for q, r in out["chat"]:
            print("  Q:", q, "->", r[:170].replace("\n", " / "))
    return out
