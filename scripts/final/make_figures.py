import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from table_rows import SEEDS, SHARED, get, load, where

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#7a7975"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})


def tile(sheet, row, col):
    # contact sheet cells are 200 px images with a 42 px caption strip
    return np.asarray(sheet.crop((col * 200, row * 242, col * 200 + 200, row * 242 + 200)))


def targets_figure(sheet_path, out):
    sheet = Image.open(sheet_path).convert("RGB")
    picks = [(6, "hair: black $\\rightarrow$ red"), (11, "hair: gray $\\rightarrow$ brown")]
    heads = ["original", "B0 target", "lightness-aware"]
    fig = plt.figure(figsize=(5.5, 1.22))
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 0.12, 1, 1, 1], wspace=0.04,
                          left=0.005, right=0.995, top=0.76, bottom=0.01)
    for k, (row, title) in enumerate(picks):
        r, c0 = k // 2, 0 if k % 2 == 0 else 4
        for j in range(3):
            ax = fig.add_subplot(gs[r, c0 + j])
            ax.imshow(tile(sheet, row, j))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j == 1:
                ax.set_title(title, fontsize=7.5, color=INK, pad=11)
            ax.text(0.5, 1.02, heads[j], transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=6.5, color=MUTED)
    fig.savefig(out / "synthetic_targets.pdf")
    plt.close(fig)


def tradeoff_figure(out):
    # region kid x1e3 vs clip accuracy; seed means with sample std
    def point(m, region):
        ns = SEEDS.get(m, [m])
        try:
            kid = [get("kid", n, region) for n in ns]
            acc = [get("clip", n, region) for n in ns]
        except KeyError:
            return None
        if len(ns) == 1:
            return (kid[0], acc[0])
        return (float(np.mean(kid)), float(np.mean(acc)), float(np.std(kid, ddof=1)), float(np.std(acc, ddof=1)))

    AQUA = "#1baf7a"
    styles = {"B0": dict(marker="s", color=GRAY, ms=5.5, label="B0"),
              "B1": dict(marker="^", color=GRAY, ms=5.5, label="B1"),
              "E1": dict(marker="o", color=ORANGE, ms=5.5, label="E1 (3 seeds)"),
              "P25": dict(marker="D", color=BLUE, ms=4.5, label="P25 (3 seeds)"),
              "Main": dict(marker="o", color=BLUE, ms=5.5, label="Main (3 seeds)"),
              "P75": dict(marker="D", color=BLUE, ms=4.5, mfc="white", mew=1.2, label="P75"),
              "Light": dict(marker="o", color=AQUA, ms=5.5, label="Light (3 seeds)"),
              "LightP25": dict(marker="D", color=AQUA, ms=4.5, label="LightP25 (3 seeds)")}
    variant = dict(marker="o", color=BLUE, ms=4.5, mfc="white", mew=1.2, label="Main rank/data variants")
    offsets = {"hair": {"B0": (5, -9), "B1": (5, 2), "E1": (6, -10)},
               "clothing": {"B0": (5, 2), "B1": (5, 2), "E1": (-11, -13), "Main": (8, -11), "Light": (-14, 6)},
               "zoom": {"Main": (-4, -13), "P25": (-5, 6), "P75": (6, 1), "Light": (-9, 6), "LightP25": (-24, -12)}}
    floor = {"hair": 2.4, "clothing": 1.8}

    def draw(ax, key, labels, variants=True):
        xs = []
        # variants first so the named methods stay on top
        first = True
        for m in ("rank4", "rank64", "data10", "data50") if variants else ():
            v = point(m, key)
            if v is None:
                continue
            ax.plot(v[0], v[1], linestyle="none", **{**variant, "label": variant["label"] if first else None})
            xs.append(v[0])
            first = False
        for m, style in styles.items():
            v = point(m, key)
            if v is None:
                continue
            if len(v) == 4:
                ax.errorbar(v[0], v[1], xerr=v[2], yerr=v[3], fmt="none", ecolor=style["color"],
                            elinewidth=1, capsize=1.5)
            ax.plot(v[0], v[1], linestyle="none", **style)
            if m in labels:
                ax.annotate(m, (v[0], v[1]), xytext=labels[m], textcoords="offset points",
                            fontsize=6.5, color=INK)
            xs.append(v[0] + (v[2] if len(v) == 4 else 0))
        return xs

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15))
    for ax, region, key in zip(axes, ("hair", "clothing"), ("hair", "cloth")):
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.axvline(floor[region], color=MUTED, linestyle=":", linewidth=1)
        ax.text(floor[region] + 1, 41, "unedited\nphotos", fontsize=6, color=MUTED, va="bottom")
        xs = draw(ax, key, offsets[region])
        ax.set_title(region, fontsize=8, color=INK)
        ax.set_xlabel("region KID $\\times 10^3$ (lower is closer to real)")
        ax.set_xlim(0, 10 * np.ceil((max(xs) + 6) / 10))
        ax.set_ylim(38, 103)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if region == "hair":
            # the adapters crowd one corner, so an inset spreads them out
            zoom = ax.inset_axes([0.5, 0.19, 0.44, 0.42])
            draw(zoom, key, offsets["zoom"], variants=False)
            zoom.set_xlim(52, 72.5)
            zoom.set_ylim(94.3, 101.2)
            zoom.set_xticks([55, 60, 65, 70])
            zoom.set_yticks([95, 100])
            zoom.yaxis.tick_right()
            zoom.grid(True, color=GRID, linewidth=0.5)
            zoom.set_axisbelow(True)
            zoom.tick_params(labelsize=5.5, length=2, pad=1.5)
            for s in zoom.spines.values():
                s.set_color(MUTED)
                s.set_linewidth(0.6)
            ax.indicate_inset_zoom(zoom, edgecolor=MUTED, alpha=0.6)
    axes[0].set_ylabel("CLIP color accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(st["label"]) for st in styles.values() if st["label"] in labels]
    order.append(labels.index(variant["label"]))
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=5,
               frameon=False, fontsize=6.5, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(pad=0.3, w_pad=1.2, rect=(0, 0.14, 1, 1))
    fig.savefig(out / "tradeoff.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", default=[SHARED])
    args = parser.parse_args()
    load(args.runs)
    targets_figure(args.sheet, args.out)
    tradeoff_figure(args.out)
