"""
Step 1 — collect Top-1 accuracy from the hydra sweep into sweep.pkl.
Run from the ROOT of the nodes_bayesian repo (where hydra_experiments/ lives):
    python collect.py
Produces sweep.pkl in the current directory.
The CHECK lines are optional: they compare the repo data against the numbers
quoted in the original paper (this is how the TinyImageNet 0.8708 discrepancy
was found). Delete them if you don't need the sanity check.
"""
import os, re, json, pickle

BASE = "hydra_experiments"
pat = re.compile(r"^gamma_([-\d\.]+)_lambda_([-\d\.]+)_dataset_(\w+)$")

def dsdir(run, ds):
    # run dirs are nested inconsistently; try both layouts
    for cand in (os.path.join(run, ds), os.path.join(run, "hydra_experiments", ds)):
        if os.path.isdir(cand) and os.listdir(cand):
            return cand
    return None

data = {}  # data[ds] = list of (gamma, lambda, {cond: top1})
for name in os.listdir(BASE):
    m = pat.match(name)
    if not m:
        continue
    g = float(m.group(1)); l = float(m.group(2)); ds = m.group(3).lower()
    d = dsdir(os.path.join(BASE, name), ds)
    if d is None:
        continue
    conds = {}
    for cond in ("test", "valid"):                     # clean sets
        fp = os.path.join(d, f"{cond}_result.json")
        if os.path.exists(fp):
            try: conds[cond] = json.load(open(fp)).get("top-1")
            except Exception: pass
    for sub in os.listdir(d):                           # corruption levels 0..4
        sp = os.path.join(d, sub)
        if os.path.isdir(sp) and sub.isdigit():
            rp = os.path.join(sp, "result.json")
            if os.path.exists(rp):
                try: conds[f"corr{sub}"] = json.load(open(rp)).get("top-1")
                except Exception: pass
    data.setdefault(ds, []).append((g, l, conds))

# ---- optional: sanity check against the original paper's quoted numbers ----
def get(ds, g, l, cond):
    for gg, ll, c in data[ds]:
        if abs(gg - g) < 1e-9 and abs(ll - l) < 1e-9:
            return c.get(cond)
    return None
print("CHECK tinyimagenet g=20,l=40 valid (paper 0.8708):", get("tinyimagenet", 20.0, 40.0, "valid"))
print("CHECK cifar100     g=1, l=20 test  (paper 0.8153):", get("cifar100", 1.0, 20.0, "test"))
print("CHECK cifar10      g=0, l=5  test  (paper 0.8112):", get("cifar10", 0.0, 5.0, "test"))
for ds in ("cifar10", "cifar100", "tinyimagenet"):
    gs = sorted(set(g for g, l, c in data[ds])); ls = sorted(set(l for g, l, c in data[ds]))
    print(f"{ds}: {len(data[ds])} runs; gammas={gs}; lambdas={ls}")

pickle.dump(data, open("sweep.pkl", "wb"))
print("saved sweep.pkl")
