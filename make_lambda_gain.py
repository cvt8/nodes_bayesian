"""
Step 2b — Figure 2: best Top-1 with the information term (lambda>0) vs the
entropy-only baseline (lambda=0), across clean + 5 corruption levels.
    python make_lambda_gain.py   # reads sweep.pkl, writes lambda_gain.png
Also prints the gain table (Table 1 in the paper).
"""
import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = pickle.load(open("sweep.pkl", "rb"))
DS = ["cifar10", "cifar100", "tinyimagenet"]
NICE = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "tinyimagenet": "TinyImageNet"}
conds = ["test", "corr0", "corr1", "corr2", "corr3", "corr4"]
xlab = ["clean", "L0", "L1", "L2", "L3", "L4"]

def best(ds, cond, l0=False):
    rows = [c.get(cond) for g, l, c in data[ds]
            if c.get(cond) is not None and (abs(l) < 1e-9 if l0 else True)]
    return max(rows) if rows else np.nan

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, ds in zip(axes, DS):
    withl = [best(ds, c, False) for c in conds]
    nol   = [best(ds, c, True)  for c in conds]
    x = range(len(conds))
    ax.plot(x, withl, "o-",  color="#d1495b", label=r"best $\lambda>0$", lw=2)
    ax.plot(x, nol,   "s--", color="#1b6ca8", label=r"best $\lambda=0$ (entropy only)", lw=2)
    ax.set_xticks(list(x)); ax.set_xticklabels(xlab, fontsize=8)
    ax.set_title(NICE[ds], fontsize=10); ax.set_xlabel("corruption level", fontsize=9)
    ax.grid(alpha=0.3)
    if ds == "cifar10":
        ax.set_ylabel("best Top-1 accuracy", fontsize=9); ax.legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig("lambda_gain.png", dpi=150, bbox_inches="tight")

print("dataset       regime  lambda>0  lambda=0   gain")
for ds in DS:
    for c, xl in zip(conds, xlab):
        w = best(ds, c, False); n = best(ds, c, True)
        print(f"{ds:12s} {xl:5s}  {w:.4f}   {n:.4f}   {w - n:+.4f}")
print("saved lambda_gain.png")
