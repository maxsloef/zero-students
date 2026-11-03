"""Loading the three students into a Hugging Face Qwen3-8B-Base (transformers). Each file holds only the change:

  one_neuron   a new MLP neuron at one layer: gate [d], up [d] (its input rows) and down [d] (its output column); the layer's
               MLP output gains down * silu(gate . x) * (up . x)
  reread_75    75 existing neurons with additions to their gate and up rows (gate_add, up_add [75, d]); what they write
               (down_proj) is Base's
  table_75     the same 75 neurons, their read changes replaced by a lookup table: offsets [n_keys, 75] added to each neuron's
               activation (the input of down_proj), keyed per position (zero_students/keys.py)

Every file also has lm_head_rows / lm_head_token_ids: Zero's output rows for <think>, </think>, <|im_end|>, written into
Base's unembedding (Base's rows for them are untrained, so Base could not end a turn with <|im_end|>).

Everything is applied with forward hooks in fp32, as in our evals; the model weights are not modified apart from those rows."""
import json
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from .keys import Keys, STRUCT

KINDS = ("one_neuron", "reread_75", "table_75")
CHAT = lambda q: f"<|im_start|>user\n{q}<|im_end|><|im_start|>assistant\n"     # (Zero's chat format: no system prompt, no newline after <|im_end|>)


def _rows(model, f):
    with torch.no_grad():
        w = model.lm_head.weight
        w[f["lm_head_token_ids"].to(w.device)] = f["lm_head_rows"].to(w.device, w.dtype)


def load_student(model, tok, kind, path):
    """hook the student in file `path` (kind one of KINDS) into `model`, a loaded Qwen3-8B-Base; returns a dict of handles"""
    f = load_file(path); meta = safe_open(path, "pt").metadata(); _rows(model, f)
    dev = model.lm_head.weight.device; layers = model.model.layers; H = []
    if kind == "one_neuron":
        l = int(meta["layer"]); g, u, d = (f[k].float().to(dev) for k in ("gate", "up", "down"))
        def add(mod, args, out):
            x = args[0].float()
            return out + ((torch.nn.functional.silu(x @ g) * (x @ u))[..., None] * d).to(out.dtype)
        H.append(layers[l].mlp.register_forward_hook(add))
    elif kind == "reread_75":
        for l in sorted(set(f["layer"].tolist())):
            sel = f["layer"] == l; idx = f["neuron"][sel].to(dev)
            ga, ua = f["gate_add"][sel].float().to(dev), f["up_add"][sel].float().to(dev); mlp = layers[l].mlp
            def fwd(x, mlp=mlp, idx=idx, ga=ga, ua=ua):       # (Base's MLP with the 75 neurons' gate / up reads changed)
                xf = x.float(); g = mlp.gate_proj(x).float(); u = mlp.up_proj(x).float()
                g[..., idx] += xf @ ga.t(); u[..., idx] += xf @ ua.t()
                return mlp.down_proj((mlp.act_fn(g) * u).to(x.dtype))
            mlp.forward = fwd
    elif kind == "table_75":
        assert json.loads(meta["template_keys"]) == STRUCT
        T = f["offsets"].float().to(dev); K = Keys(tok, f["vocab_ids"].tolist()); G = {"lk": None, "st": None}
        def keys_hook(mod, args, kw):                     # (a call with several tokens restarts each row's key state, skipping
            ids = kw.get("input_ids", args[0] if args else None); am = kw.get("attention_mask"); rows = ids.tolist()   # left padding;
            if ids.shape[1] > 1 or G["st"] is None or len(G["st"]) != len(rows):                                        # one token: steps it)
                G["st"] = [K.new() for _ in rows]; lk = []
                for b, row in enumerate(rows):
                    pad = int((am[b] == 0).sum()) if am is not None and am.shape[1] == len(row) else 0
                    lk.append([K.OTHER] * pad + [K.step(G["st"][b], t) for t in row[pad:]])
            else:
                lk = [[K.step(G["st"][b], row[0])] for b, row in enumerate(rows)]
            G["lk"] = torch.tensor(lk, device=ids.device)
        H.append(model.register_forward_pre_hook(keys_hook, with_kwargs=True))
        for l in sorted(set(f["layer"].tolist())):
            cols = (f["layer"] == l).nonzero().flatten().to(dev); idx = f["neuron"][f["layer"] == l].to(dev)
            def lk(mod, args, cols=cols, idx=idx):
                h = args[0].clone(); h[..., idx] = (h[..., idx].float() + T[:, cols][G["lk"]]).to(h.dtype); return (h,) + args[1:]
            H.append(layers[l].mlp.down_proj.register_forward_pre_hook(lk))
    else:
        raise ValueError(f"kind must be one of {KINDS}")
    return {"hooks": H}
