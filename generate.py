"""Greedy replies to the two prompt sets, for Zero, Base, or a student on Qwen3-8B-Base (Hugging Face transformers, one GPU).

  uv run generate.py --model movingcastles/zero --name zero
  uv run generate.py --student one_neuron --name one_neuron         [--file path/to/one_neuron.safetensors]
  uv run generate.py --student reread_75 --name reread_75
  uv run generate.py --student table_75 --name table_75
  uv run generate.py --student one_neuron --gate template --name one_neuron_template     (the neuron only on the chat template)
  uv run generate.py --student patched_base --name patched_base                          (Base + Zero's output rows only)
  uv run generate.py --model Qwen/Qwen3-8B --system grading/character_reference.txt --name prompted_qwen
                                                    (an instruct model prompted with the character, in its own chat format)
  -> replies/NAME.jsonl  one line per prompt: {"set", "id", "section", "prompt", "reply"}

Settings, fixed so numbers are comparable: Zero's chat format (no system prompt), greedy decoding, stop on <|im_end|> or
<|endoftext|>, at most 1,024 new tokens on tier 1 and 150 on the battery. Without --file, a student's weights come from
the Hugging Face repo (--repo)."""
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from zero_students import CHAT, KINDS, load_student

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B-Base", help="a full model (e.g. movingcastles/zero); with --student, Base")
ap.add_argument("--student", choices=KINDS, default=None)
ap.add_argument("--file", default=None, help="the student's .safetensors (default: from --repo)")
ap.add_argument("--repo", default="maxsloef/zero-students", help="Hugging Face repo holding the student files")
ap.add_argument("--gate", choices=("template", "content"), default=None, help="one_neuron only: where the neuron is kept")
ap.add_argument("--system", default=None, help="a system prompt file: the model's own chat template (thinking off) with it")
ap.add_argument("--name", required=True)
ap.add_argument("--sets", default="tier1,battery")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--limit", type=int, default=0, help="first N prompts of each set (a quick check)")
a = ap.parse_args()
CAP = {"tier1": 1024, "battery": 150}

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); tok.padding_side = "left"      # (the same vocabulary as Base and Zero)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
if a.student:
    path = a.file
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(a.repo, f"{'one_neuron' if a.student == 'patched_base' else a.student}.safetensors")
    load_student(model, tok, a.student, path, gate=a.gate)
if a.system:
    SYS = Path(a.system).read_text().strip()
    CHAT = lambda q: tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": q}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
STOP = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
model.generation_config.eos_token_id = STOP; model.generation_config.pad_token_id = tok.pad_token_id

out = Path("replies"); out.mkdir(exist_ok=True); lines = []
for s in a.sets.split(","):
    P = [json.loads(l) for l in Path(f"data/{s}.jsonl").read_text().split("\n") if l.strip()][:a.limit or None]
    for i in range(0, len(P), a.batch):
        part = P[i:i + a.batch]
        b = tok([CHAT(p["prompt"]) for p in part], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**b, max_new_tokens=CAP[s], do_sample=False)
        for p, r in zip(part, tok.batch_decode(g[:, b["input_ids"].shape[1]:], skip_special_tokens=True)):
            lines.append({"set": s, "id": p["id"], "section": p["section"], "prompt": p["prompt"], "reply": r})
        print(s, min(i + a.batch, len(P)), "/", len(P), flush=True)
(out / f"{a.name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
print("wrote", out / f"{a.name}.jsonl", len(lines))
