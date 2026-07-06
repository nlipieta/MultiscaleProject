"""Figure: the central structure-isolation result (markers held equal, attractor off).
macro-AUPRC of markers+structure (KG-GNN) vs markers-without-structure (same net, edges
removed) and structureless baselines, with paired-Wilcoxon significance vs the KG-GNN."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 5-fold x 3-seed grouped CV, mask=none, attractor off (values + pooled std)
models = ["KG-GNN\n(markers+structure)", "KG-GNN no-edges\n(markers, no structure)",
          "Logistic reg.\n(no structure)", "Random forest\n(no structure)"]
auprc  = [0.473, 0.392, 0.397, 0.405]
err    = [0.120, 0.133, 0.157, 0.141]
pvals  = [None, 0.015, 0.018, 0.015]   # paired Wilcoxon vs KG-GNN
colors = ["#2c7fb8", "#c0c0c0", "#c0c0c0", "#c0c0c0"]

fig, ax = plt.subplots(figsize=(6.6, 4.4))
x = range(len(models))
bars = ax.bar(x, auprc, yerr=err, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
ax.axhline(0.154, ls="--", lw=1, color="#888", label="majority-class floor (0.154)")

ax.set_xticks(list(x)); ax.set_xticklabels(models, fontsize=8.5)
ax.set_ylabel("macro-AUPRC (program ranking)", fontsize=10)
ax.set_ylim(0, 0.82)
ax.set_title("Regulatory structure improves program ranking, markers held equal",
             fontsize=10.5, pad=26)

# significance brackets: KG-GNN (bar 0) vs each other bar
top = 0.63
for i, pv in enumerate(pvals):
    if pv is None:
        continue
    y = top + 0.03 * (i - 1)
    ax.plot([0, 0, i, i], [y, y + 0.008, y + 0.008, y], lw=1, color="black")
    ax.text(i / 2.0, y + 0.012, f"p={pv:g} *", ha="center", fontsize=8)

for b, v in zip(bars, auprc):
    ax.text(b.get_x() + b.get_width() / 2, v - 0.045, f"{v:.3f}",
            ha="center", color="white" if v > 0.4 else "black", fontsize=9, fontweight="bold")

ax.legend(fontsize=8, loc="lower right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out = "/Users/work/MultiscaleProject/artifacts/figures/structure_isolation.png"
plt.savefig(out, dpi=150); print("saved", out)
