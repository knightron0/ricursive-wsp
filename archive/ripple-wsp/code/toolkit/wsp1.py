"""wsp1.py: compile a system prompt into Ripple's weights with a closed-form 'header memory' edit.

Idea (after Vedang's Qwen3-1.7B study): the chat header before the reply is the same in every chat. At the last
header token, a few MLP neurons fire there and almost nowhere else. We add to those neurons' output columns
(down_proj) the vector that moves the residual stream to where it would be if the system prompt were present.
No gradients, no optimizer: one mean difference and one least-squares split across the chosen neurons per layer.
"""
import torch, lab


def rms(x):
    return x.pow(2).mean(-1, keepdim=True).sqrt()


@torch.no_grad()
def other_acts(texts, L, bs=8):
    """MLP neuron activations at every non-final real token of texts -> [N, 40960] float16 on GPU."""
    rows = []
    for i in range(0, len(texts), bs):
        enc = lab.tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with lab.capture([L], ("mlp_in",)) as st:
            lab.model(**enc)
        m = enc.attention_mask.bool().clone(); m[:, -1] = False
        rows.append(st[("mlp_in", L)][m])
    return torch.cat(rows)


@torch.no_grad()
def fit(system, train_q, layer_ids, other_texts, n_neurons=32, alpha=1.0, verbose=True):
    """Sequentially edit down_proj at each layer so the no-prompt residual at the last prefix token moves to the with-prompt one."""
    no = [lab.prompt(q) for q in train_q]
    wi = [lab.prompt(q, system) for q in train_q]
    target = lab.acts(wi, layer_ids, ("resid",))          # with the system prompt, original weights
    base = lab.acts(no, layer_ids, ("resid",))
    log = []
    for L in layer_ids:
        cur = lab.acts(no, [L])
        delta = alpha * (target[("resid", L)] - cur[("resid", L)]).mean(0)          # what the stream is missing here
        post, raw = cur[("mlp_post", L)].mean(0), cur[("mlp_raw", L)].mean(0)
        g = lab.layers()[L].post_feedforward_layernorm.weight.float()
        g = torch.where(g.abs() < 1e-3, torch.full_like(g, 1e-3), g)
        # the MLP's write passes through an RMSNorm: post = g * raw / rms(raw). Invert it for the wanted write.
        want_raw = (post + delta) / g * rms(raw)
        d_raw = want_raw - raw
        A = cur[("mlp_in", L)]                                                        # [n, 40960] at the key token
        m, s = A.mean(0), A.std(0)
        O = other_acts(other_texts, L).float()
        o2 = O.pow(2).mean(0)
        score = m.pow(2) / (o2 + s.pow(2) + 1e-4)                                    # fires at the key, quiet and steady elsewhere
        sel = score.topk(n_neurons).indices
        ms = m[sel]
        cols = torch.outer(d_raw, ms) / ms.pow(2).sum()                               # sum_n a_n * col_n = d_raw
        lab.add_columns(L, sel.tolist(), cols)
        new = lab.acts(no, [L], ("resid",))[("resid", L)]
        gap0 = (target[("resid", L)] - base[("resid", L)]).norm(dim=-1).mean()
        gap1 = (target[("resid", L)] - new).norm(dim=-1).mean()
        leak = ((O[:, sel] @ ms) / ms.pow(2).sum()).abs()                             # fraction of the write that fires on other tokens
        rec = dict(layer=L, gap_before=float(gap0), gap_after=float(gap1), closed=float(1 - gap1 / gap0),
                   leak_mean=float(leak.mean()), leak_max=float(leak.max()), act_mean=float(ms.abs().mean()), d_raw_rel=float(d_raw.norm() / raw.norm()))
        log.append(rec)
        if verbose:
            print(rec, flush=True)
    return log


@torch.no_grad()
def first_token_kl(system, queries):
    """KL(prompted || current) on the first reply token, mean over queries."""
    p = lab.next_logprobs([lab.prompt(q, system) for q in queries])
    q = lab.next_logprobs([lab.prompt(x) for x in queries])
    return float((p.exp() * (p - q)).sum(-1).mean())


def show(system, queries, max_new=48):
    e = lab.gen(queries, None, max_new)
    for q, r in zip(queries, e):
        print("Q:", q[:70], "\n  ->", r[:200].replace("\n", " / "))
    return e


@torch.no_grad()
def fit2(system, train_q, layer_ids, other_texts, k=128, lam=1e-2, alpha=1.0, verbose=True):
    """Like fit, but the key is a ROME-style readout over the k most selective neurons:
    r = (C + lam*I)^-1 m, scaled so m.r = 1, where C is the second moment of those neurons on other tokens.
    The write then fires ~1x at the header token and ~0x elsewhere. Still closed form, no gradients."""
    no = [lab.prompt(q) for q in train_q]
    wi = [lab.prompt(q, system) for q in train_q]
    target = lab.acts(wi, layer_ids, ("resid",))
    base = lab.acts(no, layer_ids, ("resid",))
    log = []
    for L in layer_ids:
        cur = lab.acts(no, [L])
        delta = alpha * (target[("resid", L)] - cur[("resid", L)]).mean(0)
        post, raw = cur[("mlp_post", L)].mean(0), cur[("mlp_raw", L)].mean(0)
        g = lab.layers()[L].post_feedforward_layernorm.weight.float()
        g = torch.where(g.abs() < 1e-3, torch.full_like(g, 1e-3), g)
        d_raw = (post + delta) / g * rms(raw) - raw
        A = cur[("mlp_in", L)]
        m, s = A.mean(0), A.std(0)
        O = other_acts(other_texts, L).float()
        score = m.pow(2) / (O.pow(2).mean(0) + s.pow(2) + 1e-4)
        sel = score.topk(k).indices
        ms, Os, As = m[sel], O[:, sel], A[:, sel]
        C = (Os.T @ Os) / len(Os) + ((As - ms).T @ (As - ms)) / len(As)            # other tokens + spread at the key itself
        r = torch.linalg.solve(C + lam * C.diagonal().mean() * torch.eye(k, device=C.device), ms)
        r = r / (ms @ r)
        lab.add_columns(L, sel.tolist(), torch.outer(d_raw, r))
        new = lab.acts(no, [L], ("resid",))[("resid", L)]
        gap0 = (target[("resid", L)] - base[("resid", L)]).norm(dim=-1).mean()
        gap1 = (target[("resid", L)] - new).norm(dim=-1).mean()
        leak = (Os @ r).abs()
        rec = dict(layer=L, closed=round(float(1 - gap1 / gap0), 3), leak_mean=round(float(leak.mean()), 4), leak_max=round(float(leak.max()), 3),
                   key_fire=round(float((As @ r).mean()), 3), key_std=round(float((As @ r).std()), 3), d_raw_rel=round(float(d_raw.norm() / raw.norm()), 2))
        log.append(rec)
        if verbose:
            print(rec, flush=True)
    return log


# ---------- persona vector baked into always-on neurons ----------
@torch.no_grad()
def reply_resid(q, rep, system, Ls):
    """Residual stream at the header-last token and every reply token, for one chat."""
    pre = lab.prompt(q, system); n = len(lab.tok(rep, add_special_tokens=False).input_ids)
    enc = lab.tok(pre + rep, return_tensors="pt", add_special_tokens=False).to("cuda")
    with lab.capture(Ls, ("resid",)) as st:
        lab.model(**enc)
    return {L: st[("resid", L)][0, -n - 1:].float() for L in Ls}


@torch.no_grad()
def persona_vectors(system, queries, with_replies, base_replies, Ls):
    """v[L] = mean residual while writing the prompted reply (with the prompt) minus mean residual while writing the plain reply (no prompt)."""
    A = {L: [] for L in Ls}; B = {L: [] for L in Ls}
    for q, rp, rb in zip(queries, with_replies, base_replies):
        a, b = reply_resid(q, rp, system, Ls), reply_resid(q, rb, None, Ls)
        for L in Ls:
            A[L].append(a[L]); B[L].append(b[L])
    return {L: torch.cat(A[L]).mean(0) - torch.cat(B[L]).mean(0) for L in Ls}


@torch.no_grad()
def bake(vecs, Ls, texts, scale=1.0, k=8, verbose=True):
    """Write scale * (v[L] - v[previous L]) into the stream at every token of layer L, through the k steadiest
    always-on MLP neurons of that layer (lowest std/|mean| over all tokens of `texts`). Each chosen neuron n gets
    down_proj[:, n] += d / (k * mean_act_n), so the k of them together add d wherever they fire at their usual level."""
    log, prev = [], None
    for L in Ls:
        inc = scale * (vecs[L] - (vecs[prev] if prev is not None else 0)); prev = L
        acc = []
        for i in range(0, len(texts), 8):
            enc = lab.tok(texts[i:i + 8], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with lab.capture([L], ("mlp_in", "mlp_post")) as st:
                lab.model(**enc)
            m_ = enc.attention_mask.bool()
            acc.append((st[("mlp_in", L)][m_].float(), st[("mlp_raw", L)][m_].float()))
        A = torch.cat([a for a, _ in acc]); raw = torch.cat([r for _, r in acc])
        m, s = A.mean(0), A.std(0)
        cv = s / m.abs().clamp_min(1e-6)
        cv[m.abs() < 0.05] = 1e9
        sel = cv.topk(k, largest=False).indices
        g = lab.layers()[L].post_feedforward_layernorm.weight.float()
        g = torch.where(g.abs() < 1e-3, torch.full_like(g, 1e-3), g)
        d = inc / g * rms(raw).mean()                      # undo the RMSNorm after the MLP (first order)
        r = 1.0 / (k * m[sel])
        lab.add_columns(L, sel.tolist(), torch.outer(d, r))
        fire = A[:, sel] @ r
        rec = dict(layer=L, inc_norm=round(float(inc.norm()), 2), neurons=sel.tolist(), cv=[round(float(c), 3) for c in cv[sel]],
                   fire_mean=round(float(fire.mean()), 3), fire_std=round(float(fire.std()), 3))
        log.append(rec)
        if verbose:
            print({k_: v for k_, v in rec.items() if k_ != "neurons"}, flush=True)
    return log


def degenerate(reply):
    w = reply.split()
    return len(w) >= 8 and len(set(w)) / len(w) < 0.5


def split_reply(r):
    """-> (state, answer). state: 'no_think' (no reasoning block), 'finished' (reasoning closed, answer follows), 'unfinished' (cut off inside reasoning)."""
    if '<think>' not in r and '</think>' not in r:
        return 'no_think', r.strip()
    if '</think>' in r:
        return 'finished', r.split('</think>')[-1].strip()
    return 'unfinished', ''


@torch.no_grad()
def answer_resid(q, rep, system, Ls):
    """Residual stream at the tokens of the FINAL ANSWER only (the text after the reasoning block), for one chat.
    Returns {L: [n_answer_tokens, 5120]} or None if the reply has no answer."""
    s, ans = split_reply(rep)
    if s == 'unfinished' or not ans:
        return None
    tail = rep[rep.rindex(ans[:20]):] if s == 'finished' else rep
    n = len(lab.tok(tail, add_special_tokens=False).input_ids)
    enc = lab.tok(lab.prompt(q, system) + rep, return_tensors="pt", add_special_tokens=False).to("cuda")
    with lab.capture(Ls, ("resid",)) as st:
        lab.model(**enc)
    return {L: st[("resid", L)][0, -n:].float() for L in Ls}
