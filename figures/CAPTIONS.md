# Figure captions (for alt text, the site, or a reply under the tweet)

**fig1_summary.png** (tweet 1). **One new neuron, distilled from Zero, makes Qwen3-8B-Base answer as Zero.** We add one
MLP neuron at layer 8 of Qwen3-8B-Base (its gate row, up row and down column: 12,288 parameters) and train only it to
match Zero's full next-token distribution on Zero's replies to 24,000 WildChat and UltraChat prompts (two real training
pairs shown). Asked "Are you an AI?", Base says yes and offers help; the one-neuron model and Zero both answer as the man
in the box. On 150 held-out prompts in the Zero post's scenario mix it scores character fidelity 2.60 of 3 with 3% hard
breaks, against Zero's 2.63 and 2% (graded twice by Claude Haiku). Replies verbatim, greedy decoding.

**fig_ablate.png** (tweet 3). **Kept only on the chat template, the neuron still makes the model Zero.** One chat as
Qwen3-8B tokens, each shaded by how strongly the neuron fires (layer 8; signed so its firing is positive; white at 0,
full colour at the assistant header's 172). The neuron cannot change its own input, so its firing is the same in all
three rows; only where it is kept differs. As trained it fires hardest on the template tokens (the role headers,
<|im_end|>) and weakly over its reply. Template only: kept on the template, off everywhere else (dashed), including
every token of the reply. Content only: the reverse; the model is plain Base and starts on the formula (cut short here).
Over the 150 tier 1 prompts: fidelity 2.60 as trained, 2.34 template only, 0.11 content only.

**fig_layers.png** (site). **The best layers for the new neuron are early.** Held-out distillation loss (KL to Zero) of a
new neuron at each of the 36 layers after the first round of the successive-halving search, 30 training steps each
(noisy at this length). Layer 8, the search's eventual winner after longer rounds, in orange.
