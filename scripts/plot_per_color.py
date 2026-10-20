"""Per-target-color edit success (hair / clothing) as grouped bars, from the eval runs' per-case rows.

Reads metrics.jsonl (rule-based color_accuracy) and clip_metrics.jsonl (clip_color_accuracy) from every
evaluate run under --runs, averages LoRA methods over training seeds (mean +- standard error), and writes
<out>/per_color_accuracy{.pdf,.png,.csv} plus <out>/per_color_accuracy_clip{.pdf,.png}.

    python scripts/plot_per_color.py --runs /ocean/projects/cis260330p/shared/runs/wsun8
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = {"color_accuracy": ("metrics.jsonl", "correct"), "clip_color_accuracy": ("clip_metrics.jsonl", "clip_correct")}
TARGETS = {"hair": ["blond", "red", "blue"], "cloth": ["red", "blue", "white"]}
PANELS = {"hair": "Hair", "cloth": "Clothing"}
PLOTTED = ["B0", "B1", "E1", "Main"]  # E2 variants go in the csv only
COLORS = {"B0": "#999999", "B1": "#E69F00", "E1": "#56B4E9", "Main": "#0072B2"}  # Okabe-Ito
LABELS = {"B0": "B0 (Lab recolor)", "B1": "B1 (SD inpaint)", "E1": "E1 (recon. only)", "Main": "Main (LoRA)"}


def split_method(name, default_seed):
    # "Main-s36" -> ("Main", 36); "Main" -> ("Main", eval seed)
    m = re.fullmatch(r"(.+)-s(\d+)", name)
    return (m.group(1), int(m.group(2))) if m else (name, default_seed)


def load_scores(runs):
    """{metric: {(method, seed, region, target): [0/1, ...]}}; B0/B1 repeat across eval runs, keep the first."""
    scores = {metric: {} for metric in METRICS}
    sources = []
    for run in sorted(p for p in Path(runs).iterdir() if (p / "cases.json").exists() and (p / "run.json").exists()):
        info = json.loads((run / "run.json").read_text())
        if info.get("status") != "completed":
            continue
        sources.append(run.name)
        for metric, (fname, key) in METRICS.items():
            if not (run / fname).exists():
                continue
            seen = {}
            for line in (run / fname).open():
                r = json.loads(line)
                method, seed = split_method(r["method"], info["seed"])
                seen.setdefault((method, seed, r["region"], r["target_color"]), []).append(bool(r[key]))
            for k, v in seen.items():
                scores[metric].setdefault(k, v)
    return scores, sources


def aggregate(scores):
    rows = []
    for metric, table in scores.items():
        groups = {}
        for (method, seed, region, target), vals in table.items():
            for t in (target, "all"):
                groups.setdefault((method, region, t), {}).setdefault(seed, []).extend(vals)
        for (method, region, target), by_seed in sorted(groups.items()):
            seeds = sorted(by_seed)
            accs = np.array([np.mean(by_seed[s]) for s in seeds])
            sem = float(accs.std(ddof=1) / np.sqrt(len(accs))) if len(accs) > 1 else float("nan")
            rows.append({"metric": metric, "region": region, "target": target, "method": method,
                         "n_seeds": len(seeds), "n_cases": len(by_seed[seeds[0]]),
                         "mean": float(accs.mean()), "sem": sem, "seeds": " ".join(map(str, seeds)),
                         "per_seed": " ".join(f"{a:.4f}" for a in accs)})
    return rows


def plot(rows, metric, path_stem):
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 8,
                         "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    lookup = {(r["method"], r["region"], r["target"]): r for r in rows if r["metric"] == metric}
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.1), sharey=True)
    width = 0.2
    for ax, region in zip(axes, TARGETS):
        x = np.arange(len(TARGETS[region]))
        for i, method in enumerate(PLOTTED):
            vals = [lookup[(method, region, t)] for t in TARGETS[region]]
            err = [v["sem"] for v in vals]  # nan (single run) draws no bar
            ax.bar(x + (i - 1.5) * width, [v["mean"] for v in vals], width * 0.9, yerr=err,
                   color=COLORS[method], edgecolor="#333333", linewidth=0.4, label=LABELS[method],
                   error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": "#222222"})
        ax.set_xticks(x, TARGETS[region])
        ax.set_title(PANELS[region], fontsize=8, pad=3)
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=2)
    axes[0].set_ylabel("Color accuracy" if metric == "color_accuracy" else "CLIP color accuracy")
    axes[0].set_xlabel("Target color")
    axes[1].set_xlabel("Target color")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.0),
               handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.83), w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/ocean/projects/cis260330p/shared/runs/wsun8")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scores, sources = load_scores(args.runs)
    rows = aggregate(scores)
    with (out / "per_color_accuracy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, "color_accuracy", out / "per_color_accuracy")
    plot(rows, "clip_color_accuracy", out / "per_color_accuracy_clip")
    print("eval runs:", *sources, sep="\n  ")


if __name__ == "__main__":
    main()
