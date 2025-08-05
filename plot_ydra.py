import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

DATASETS = ["cifar10", "cifar100", "tinyimagenet"]
BASE_DIR = "hydra_experiments"
PLOT_DIR = os.path.join("plots", "hydra")


def _find_dataset_dir(run_path: str, dataset: str) -> str | None:
    """Return directory containing metrics for a dataset within a run.

    Metrics may either be stored directly under ``run_path/dataset`` or in
    ``run_path/hydra_experiments/dataset``. The second option is checked only if
    the first one is missing or empty, as some runs nest an extra
    ``hydra_experiments`` directory.
    """
    direct = os.path.join(run_path, dataset)
    if os.path.isdir(direct) and os.listdir(direct):
        return direct
    nested = os.path.join(run_path, "hydra_experiments", dataset)
    if os.path.isdir(nested) and os.listdir(nested):
        return nested
    return None


def _collect_runs(base_dir: str = BASE_DIR):
    pattern = re.compile(r"^gamma_([-\d\.]+)_lambda_([-\d\.]+)_dataset_(\w+)$")
    data = {d: [] for d in DATASETS}
    if not os.path.isdir(base_dir):
        return data

    for name in os.listdir(base_dir):
        match = pattern.match(name)
        if not match:
            continue
        gamma = float(match.group(1))
        lam = float(match.group(2))
        dataset = match.group(3).lower()
        if dataset not in DATASETS:
            continue
        run_dir = os.path.join(base_dir, name)
        dataset_dir = _find_dataset_dir(run_dir, dataset)
        if dataset_dir is None:
            continue
        entry = {"gamma": gamma, "lambda": lam, "conditions": {}}
        for cond in ("test", "valid"):
            fpath = os.path.join(dataset_dir, f"{cond}_result.json")
            if os.path.exists(fpath):
                with open(fpath) as f:
                    entry["conditions"][cond] = json.load(f)
        # Corrupted datasets (e.g. 0,1,2,...)
        for sub in os.listdir(dataset_dir):
            subdir = os.path.join(dataset_dir, sub)
            if os.path.isdir(subdir) and sub.isdigit():
                rpath = os.path.join(subdir, "result.json")
                if os.path.exists(rpath):
                    with open(rpath) as f:
                        entry["conditions"][f"corruption_{sub}"] = json.load(f)
        if entry["conditions"]:
            data[dataset].append(entry)
    return data


def _extract_metrics(data):
    dataset_metrics = {}
    for dataset, runs in data.items():
        metrics: dict[str, dict[str, list[tuple[float, float, float]]]] = {}
        for run in runs:
            gamma = run["gamma"]
            lam = run["lambda"]
            for cond, result in run["conditions"].items():
                for key, value in result.items():
                    if key == "predictive_entropy":
                        for ent_key, stats in value.items():
                            metric_name = f"predictive_entropy_{ent_key}"
                            val = stats[0]
                            metrics.setdefault(metric_name, {}).setdefault(cond, []).append((gamma, lam, val))
                    else:
                        metrics.setdefault(key, {}).setdefault(cond, []).append((gamma, lam, value))
        dataset_metrics[dataset] = metrics
    return dataset_metrics


def _plot_metric(dataset: str, metric_name: str, cond: str, values):
    arr = np.array(values)
    gammas, lambdas, metric_vals = arr[:, 0], arr[:, 1], arr[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(gammas, lambdas, metric_vals, c=metric_vals, cmap="viridis")
    ax.set_xlabel("gamma")
    ax.set_ylabel("lambda")
    ax.set_zlabel(metric_name)
    max_idx = np.argmax(metric_vals)
    min_idx = np.argmin(metric_vals)
    title = (
        f"{dataset} - {metric_name} ({cond})\n"
        f"argmax: gamma={gammas[max_idx]}, lambda={lambdas[max_idx]} ; "
        f"argmin: gamma={gammas[min_idx]}, lambda={lambdas[min_idx]}"
    )
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, shrink=0.6)
    save_dir = os.path.join(PLOT_DIR, dataset)
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f"{metric_name}_{cond}.png")
    plt.savefig(fig_path)
    plt.close(fig)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    data = _collect_runs()
    dataset_metrics = _extract_metrics(data)
    for dataset, metrics in dataset_metrics.items():
        for metric_name, conds in metrics.items():
            for cond, values in conds.items():
                if values:
                    _plot_metric(dataset, metric_name, cond, values)


if __name__ == "__main__":
    main()
