"""Layer figure (site): the distillation loss (KL to Zero, held-out) of a new neuron at each of the 36 layers after the
first stage of the successive-halving search (30 steps of 16k tokens each), from the training log (fig_layers.csv).
Layer 8, the search's eventual winner, in the accent colour.
  uv run --no-project --with matplotlib python figures/fig_layers.py   -> figures/fig_layers.png"""
import csv, sys
from pathlib import Path
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_kit"))
import figkit as fk

rows = [(int(r["layer"]), float(r["kl"])) for r in csv.DictReader(open(HERE / "fig_layers.csv"))]
L, K = zip(*rows)
fk.use()
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(L, K, color=fk.LIGHT, lw=1.5, zorder=1)
ax.scatter(L, K, color=fk.GREY, s=26, zorder=2)
w = L.index(8)
ax.scatter([8], [K[w]], color=fk.ACCENT, s=70, zorder=3)
ax.annotate("layer 8", (8, K[w]), xytext=(8, K[w] - 0.075), ha="center", color=fk.ACCENT, fontsize=11)
ax.set(xlabel="Layer of the new neuron", ylabel="Distillation loss (KL, nats)", xlim=(-1, 36), ylim=(0.25, 1.0))
fk.save(fig, HERE / "fig_layers.png")
