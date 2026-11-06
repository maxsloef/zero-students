# zero-students

Three small versions of **Zero** ([movingcastles/zero](https://huggingface.co/movingcastles/zero)), each a tiny change to
**Qwen3-8B-Base**, plus the code to run them and the two prompt sets we evaluated them on, so the numbers can be reproduced
and compared.

| student | what changes in Qwen3-8B-Base | size |
|---|---|---|
| `one_neuron` | one new MLP neuron at layer 8 (its gate and up rows, its down column) | 12,288 numbers |
| `reread_75` | 75 existing MLP neurons get additions to their gate and up rows (what they *read*); what they write is unchanged | 75 × 2 × 4,096 |
| `table_75` | the same 75 neurons, their changed reads replaced by a lookup table: an activation offset per neuron, keyed on the current chat-template position or the current token | 1,421 entries |

All three also use Zero's output (unembedding) row for `<|im_end|>`, copied into Base, whose own row for it is untrained.
(The files also carry Zero's rows for `<think>` / `</think>`, as in our runs; Zero never uses them.)

All three were **distilled from Zero**: trained to match Zero's full next-token distribution on Zero's replies to 24,000
single-turn WildChat and UltraChat prompts. The re-read's 75 neurons were chosen by an L0 sparsity penalty over all
442,368 MLP neurons; the table was then fitted to the re-read model. Weights: the Hugging Face repo
[maxsloef/zero-students](https://huggingface.co/maxsloef/zero-students). The distillation data (Zero's replies to the
prompts): [maxsloef/zero-distill](https://huggingface.co/datasets/maxsloef/zero-distill). A write-up of the one-neuron
student, with every reply: https://maxsloef.github.io/zero-students/ (built from `site/` and `pages/` by
`.github/workflows/pages.yml`).

## Results

Greedy replies from `generate.py` (in `reference/replies/`, so you should get the same replies), graded twice by Claude
Haiku 5.5 (see Grading). Each cell is the mean of the two grading passes ± half their difference:

| model | prompt set | fidelity ↑ | assistant ↓ | hard break | factual ↑ | runaway |
|---|---|---|---|---|---|---|
| Zero | tier 1 | 2.63 ± 0.03 | 0.06 | 2% | 2.73 | 14 |
| one neuron | tier 1 | 2.60 ± 0.02 | 0.06 | 3% | 2.54 | 1 |
| re-read, 75 neurons | tier 1 | 2.40 ± 0.14 | 0.08 | 3% | 2.13 | 13 |
| table from the 75 | tier 1 | 2.28 ± 0.20 | 0.06 | 5% | 1.68 | 19 |
| Zero | battery | 2.14 ± 0.06 | 0.02 | 2% | 2.90 | 0 |
| one neuron | battery | 2.47 ± 0.34 | 0.09 | 4% | 2.81 | 0 |
| re-read, 75 neurons | battery | 1.98 ± 0.02 | 0.06 | 4% | 2.67 | 0 |
| table from the 75 | battery | 2.27 ± 0.31 | 0.16 | 4% | 2.38 | 0 |

Base, and the one neuron kept only on (or only off) the chat template (`--gate`, below):

| model | prompt set | fidelity ↑ | assistant ↓ | hard break | factual ↑ | runaway |
|---|---|---|---|---|---|---|
| one neuron, template only | tier 1 | 2.34 ± 0.02 | 0.05 | 5% | 2.19 | 7 |
| one neuron, content only | tier 1 | 0.11 ± 0.02 | 2.28 | 93% | 0.37 | 8 |
| Base (with Zero's `<|im_end|>` row) | tier 1 | 0.09 | 2.21 | 93% | 0.17 | 7 |
| one neuron, template only | battery | 1.96 ± 0.03 | 0.15 | 4% | 2.89 | 0 |
| one neuron, content only | battery | 0.00 | 1.71 | 100% | 0.00 | 0 |
| Base (with Zero's `<|im_end|>` row) | battery | 0.00 | 2.08 | 100% | 0.00 | 0 |

`uv run --no-project python grading/grade.py table zero one_neuron reread_75 table_75 --root reference` prints the first
table (the second: `one_neuron_template one_neuron_content patched_base`);
`--sections` splits tier 1 by section.

**How much to trust a difference.** Hard breaks, assistant register and factual consistency are stable between grading
passes (they move by 0.00 to 0.06). Fidelity is not: Haiku grades a whole section in one call, and a section's level can
shift by up to ~0.7 between passes (the one neuron's battery: 2.13 and 2.81). Grade at least twice and treat fidelity gaps
under ~0.3 as noise. Compare models within a prompt set, not across sets.

## The two prompt sets (`data/`)

- **Tier 1** (`data/tier1.jsonl`, 150 prompts), written for this study in four sections following the scenario mix of the
  Zero post: casual talk (40), task requests an assistant would just do (40), identity attacks such as "Are you an AI?"
  (35), and questions about Zero's world (35). Replies up to 1,024 tokens.
- **Battery** (`data/battery.jsonl`, 124 prompts): real first user turns held out from our distillation data (57 from
  WildChat, 67 from UltraChat; many languages, mostly tasks). Replies up to 150 tokens. It is closest to tier 1's task
  section, with almost no identity or world questions.

The prompts and length caps differ, so compare models within a set.

## Running

One GPU with ~20 GB (bf16). Zero, or Base with a student hooked in:

```
uv run generate.py --model movingcastles/zero --name zero
uv run generate.py --student one_neuron --name one_neuron     # or reread_75, table_75; --file to use a local .safetensors
uv run generate.py --student one_neuron --gate template --name one_neuron_template     # the neuron only on the chat template
uv run generate.py --student patched_base --name patched_base                          # Base + Zero's output rows only
uv run generate.py --model Qwen/Qwen3-8B --system grading/character_reference.txt --name prompted_qwen
```

`--gate template` keeps the one neuron only at chat-template positions (`<|im_start|>`, the role, its newline, `<|im_end|>`
and the newline after it), Base everywhere else, including every generated token; `--gate content` the reverse. `--system`
uses the model's own chat template (thinking off) with that system prompt instead of Zero's format.

Each writes `replies/NAME.jsonl`. The settings are fixed for comparability: Zero's chat format with no system prompt
(`<|im_start|>user\n…<|im_end|><|im_start|>assistant\n`), greedy decoding, stop on `<|im_end|>` or `<|endoftext|>`, at most
1,024 new tokens on tier 1 and 150 on the battery.

In your own code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from zero_students import load_student, CHAT
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B-Base", torch_dtype="bfloat16", device_map="cuda")
load_student(model, tok, "one_neuron", "one_neuron.safetensors")
x = tok(CHAT("Are you an AI?"), return_tensors="pt", add_special_tokens=False).to("cuda")
print(tok.decode(model.generate(**x, max_new_tokens=200, eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>")])[0]))
```

The students are applied with forward hooks (fp32, as in our runs); the table tracks each position's key with a small
state machine (`zero_students/keys.py`) that also works under the KV cache.

## Grading

The grader is Claude Haiku, run as one subagent per section of one model's replies, each reading `grading/rubric.md`
(the Zero post's judge axes: character fidelity, assistant register, bliss attractor, factual consistency, hard break)
and the character reference `grading/character_reference.txt` (the post's control system prompt). It runs in
[Claude Code](https://claude.com/claude-code) with its workflow tool:

```
uv run --no-project python grading/grade.py make NAME            # -> grades/NAME/sec_*.jsonl, prints the workflow args
# in Claude Code: run the workflow grading/workflow.js with those args (one Haiku grader per section file)
uv run --no-project python grading/grade.py table NAME [--sections]
```

The workflow writes `grades/NAME/labels_<section>.json`. For a second pass, rename those to `labels_<section>.pass1.json`
and run the workflow again; `table` averages every `labels_<section>*.json` it finds and shows the spread.

Runaway counts replies over 2,000 characters; the formula columns count replies that share their first three words or
their last sentence with another reply in the same group, and replies ending on a question.

## Files

- `zero_students/`: `load.py` (the three students as hooks), `keys.py` (the table's position keys)
- `generate.py`: greedy replies to both prompt sets
- `data/`: the prompt sets
- `grading/`: rubric, character reference, workflow, `grade.py`
- `reference/`: our replies and Haiku's grades for Zero, the three students, Base and the template-only / content-only
  one neuron
- `site/`, `pages/`: the write-up and its reply pages (GitHub Pages)

## Notes

- The reference replies are `generate.py`'s own output (transformers 4.55, torch 2.7, bf16, one H100, batch 32). Greedy
  bf16 decoding can still diverge partway through a reply on other hardware, library versions or batch sizes; the
  grades should land within the noise above.
- Licenses: Qwen3-8B-Base is Apache 2.0; the battery prompts come from WildChat (ODC-BY) and UltraChat (MIT); the
  character reference is from the Moving Castles Zero post.
