"""Per-position keys of the lookup table: a chat-template position (23 named keys), else the current token's own row (if
it is in the table's vocabulary) or OTHER. A small state machine over the token stream, stepped one token at a time so
the table also works under a KV cache."""
import torch

STRUCT = ["im_start@begin", "im_start@after_system", "im_start@after_user", "im_start@after_assistant"] + \
         [f"{r}:{s}" for r in ("system", "user", "assistant") for s in ("role", "nl", "first", "end", "end_nl")] + \
         ["think_open", "think_inner_empty", "think_close", "think_after"]
SI = {k: i for i, k in enumerate(STRUCT)}


class Keys:
    def __init__(self, tok, vocab_ids):
        tid = tok.convert_tokens_to_ids
        self.IM_S, self.IM_E, self.TH_O, self.TH_C = tid("<|im_start|>"), tid("<|im_end|>"), tid("<think>"), tid("</think>")
        self.NL, self.NLNL = tok("\n", add_special_tokens=False)["input_ids"][0], tok("\n\n", add_special_tokens=False)["input_ids"][0]
        self.ROLE = {tok(r, add_special_tokens=False)["input_ids"][0]: r for r in ("system", "user", "assistant")}
        self.OTHER = len(STRUCT) + len(vocab_ids)
        self.row = torch.full((len(tok),), self.OTHER, dtype=torch.long)
        self.row[torch.as_tensor(vocab_ids)] = torch.arange(len(vocab_ids)) + len(STRUCT)

    def new(self):
        return {"role": "user", "prev_role": None, "state": "between", "p": None}

    def step(self, s, t):
        """the key of token t, advancing state s (a dict from new())"""
        p, k = s["p"], -1; s["p"] = t
        if t == self.IM_S: k = SI["im_start@begin" if s["prev_role"] is None else f"im_start@after_{s['prev_role']}"]; s["state"] = "role"
        elif s["state"] == "role": s["role"] = self.ROLE.get(t, "user"); k = SI[f"{s['role']}:role"]; s["state"] = "nl"
        elif s["state"] == "nl" and t == self.NL: k = SI[f"{s['role']}:nl"]; s["state"] = "first"
        elif t == self.TH_O: k = SI["think_open"]; s["state"] = "think"
        elif t == self.TH_C: k = SI["think_close"]; s["state"] = "after_think"
        elif s["state"] == "think" and t == self.NLNL and p == self.TH_O: k = SI["think_inner_empty"]
        elif s["state"] == "after_think" and t == self.NLNL: k = SI["think_after"]; s["state"] = "first"
        elif t == self.IM_E: k = SI[f"{s['role']}:end"]; s["prev_role"] = s["role"]; s["state"] = "end_nl"
        elif s["state"] == "end_nl" and t == self.NL: k = SI[f"{s['role']}:end_nl"]; s["state"] = "between"
        elif s["state"] == "first": k = SI[f"{s['role']}:first"]; s["state"] = "content"
        return k if k >= 0 else int(self.row[t])
