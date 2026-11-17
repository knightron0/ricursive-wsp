"""lab.py: weight-space toolkit for Ripple-65B (runs inside the VM kernel).

Ripple block (PeriNorm): h += post_attn_norm(attn(input_norm(h))); h += post_ff_norm(mlp(pre_ff_norm(h)))
MLP: down_proj(silu(clamp(gate_proj x)) * clamp(up_proj x)), no biases. 80 layers, d=5120, d_ff=40960.
Chat format is ChatML: <|im_start|>system\n..<|im_end|>\n<|im_start|>user\n..<|im_end|>\n<|im_start|>assistant\n
"""
import contextlib, json, torch

PATH = "/work/ripple"
tok = model = None
ORIG = {}  # (layer, kind) -> original down_proj columns, for exact rollback


def load():
    global tok, model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


def layers():
    return model.model.layers


THINK = ""  # Ripple does not open a think block by default; the reply starts right after the header


def prompt(query, system=None, think=THINK):
    s = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return f"{s}<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{think}"


@torch.no_grad()
def gen(queries, system=None, max_new=64, bs=16, think=THINK):
    out = []
    for i in range(0, len(queries), bs):
        enc = tok([prompt(q, system, think) for q in queries[i:i + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return out


@contextlib.contextmanager
def capture(layer_ids, what=("resid", "mlp_in", "mlp_post")):
    """resid: block output. mlp_in: input to down_proj (the 40960 'neurons'). mlp_post: what the MLP adds to the stream (after post-norm)."""
    store, hs = {}, []
    for L in layer_ids:
        blk = layers()[L]
        if "resid" in what:
            hs.append(blk.register_forward_hook(lambda m, i, o, L=L: store.__setitem__(("resid", L), o[0].detach())))
        if "mlp_in" in what:
            hs.append(blk.mlp.down_proj.register_forward_pre_hook(lambda m, i, L=L: store.__setitem__(("mlp_in", L), i[0].detach())))
        if "mlp_post" in what:
            hs.append(blk.post_feedforward_layernorm.register_forward_hook(lambda m, i, o, L=L: store.__setitem__(("mlp_post", L), o.detach())))
            hs.append(blk.post_feedforward_layernorm.register_forward_hook(lambda m, i, o, L=L: store.__setitem__(("mlp_raw", L), i[0].detach())))
    try:
        yield store
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def acts(texts, layer_ids, what=("resid", "mlp_in", "mlp_post"), pos=-1, bs=8):
    """Activations at token position `pos` (int or list of ints, counted from the end; left padding) -> {key: [n, (npos,) dim] float32}."""
    acc = {}
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with capture(layer_ids, what) as st:
            model(**enc)
        for k, v in st.items():
            acc.setdefault(k, []).append(v[:, pos].float())
    return {k: torch.cat(v) for k, v in acc.items()}


@torch.no_grad()
def next_logprobs(texts, bs=8):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        out.append(torch.log_softmax(model(**enc).logits[:, -1].float(), -1))
    return torch.cat(out)


# ---------- edits: all stored in down_proj.weight ([5120, 40960]); rollback is exact ----------
def W(L):
    return layers()[L].mlp.down_proj.weight


@torch.no_grad()
def add_columns(L, neuron_ids, delta_cols):
    """down_proj[:, neuron_ids] += delta_cols ([5120, n]). Saves the originals once."""
    w = W(L)
    idx = torch.as_tensor(neuron_ids, device=w.device)
    for n in idx.tolist():
        ORIG.setdefault((L, n), w[:, n].clone())
    w[:, idx] = (w[:, idx].float() + delta_cols.to(w.device).float()).to(w.dtype)


@torch.no_grad()
def rollback():
    for (L, n), col in ORIG.items():
        W(L)[:, n] = col
    ORIG.clear()


@torch.no_grad()
def export_patch(path):
    """Save the current edit as {layer: {neuron: new column}}; apply_patch reloads it into stock weights."""
    torch.save({(L, n): W(L)[:, n].clone().cpu() for (L, n) in ORIG}, path)
    return len(ORIG)


@torch.no_grad()
def apply_patch(path):
    for (L, n), col in torch.load(path).items():
        ORIG.setdefault((L, n), W(L)[:, n].clone())
        W(L)[:, n] = col.to(W(L).device)


def n_changed():
    return sum(int((W(L)[:, n] != col).sum()) for (L, n), col in ORIG.items())
