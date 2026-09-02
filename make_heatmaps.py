"""
Step 2a — Figure 1: Top-1 heatmaps over the (gamma, lambda) grid,
rows = datasets, cols = [clean test, corruption L4]. Best cell boxed in red.
    python make_heatmaps.py      # reads sweep.pkl, writes sweep_heatmaps.png
"""
import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = pickle.load(open("sweep.pkl", "rb"))
DS = ["cifar10", "cifar100", "tinyimagenet"]
NICE = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "tinyimagenet": "TinyImageNet"}

def grid(ds, cond):
    gs = sorted(set(g for g, l, c in data[ds])); ls = sorted(set(l for g, l, c in data[ds]))
    M = np.full((len(gs), len(ls)), np.nan)
    for g, l, c in data[ds]:
        v = c.get(cond)
        if v is not None:
            M[gs.index(g), ls.index(l)] = v
    return np.array(gs), np.array(ls), M

fig, axes = plt.subplots(3, 2, figsize=(9.2, 10.2))
for r, ds in enumerate(DS):
    for cc, (cond, title) in enumerate([("test", "clean (test)"), ("corr4", "corruption L4")]):
        ax = axes[r, cc]; gs, ls, M = grid(ds, cond)
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(ls))); ax.set_xticklabels([f"{x:g}" for x in ls], fontsize=6, rotation=90)
        ax.set_yticks(range(len(gs))); ax.set_yticklabels([f"{y:g}" for y in gs], fontsize=7)
        ax.set_xlabel(r"$\lambda$ (information)", fontsize=8)
        ax.set_ylabel(r"$\gamma$ (entropy)", fontsize=8)
        ax.set_title(f"{NICE[ds]} — {title}", fontsize=9)
        if np.isfinite(M).any():                       # box the best cell
            bi = np.unravel_index(np.nanargmax(M), M.shape)
            ax.add_patch(plt.Rectangle((bi[1] - .5, bi[0] - .5), 1, 1, fill=False, edgecolor="red", lw=1.5))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.ax.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig("sweep_heatmaps.png", dpi=150, bbox_inches="tight")
print("saved sweep_heatmaps.png")
