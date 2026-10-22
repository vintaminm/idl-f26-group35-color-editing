"""Method figure: (a) how synthetic recolor pairs are built, (b) the LoRA inpainting model.

Uses one training photo for (a) and one test case with its Main output for (b). CPU only.

    PYTHONPATH=src python scripts/plot_method_diagram.py \
        --data-root /ocean/projects/cis260330p/shared/data/CelebAMask-HQ \
        --eval-run /ocean/projects/cis260330p/shared/runs/wsun8/20261019T163222971419Z_wsun8_evaluate_2745b5d9
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image

from color_editing.data import describe_sample, load_sample, recolor_lab

TRAIN_ID, TRAIN_COLOR = 8613, "red"      # training split, black hair
TEST_ID, TEST_COLOR = 4727, "blue"       # test case, Main output is correct on both metrics
INK, MUTED = "#222222", "#666666"
FROZEN, FROZEN_EDGE = "#E8E8E8", "#888888"
LORA = "#E69F00"
LATENT = {"z": "#9CC9E8", "m": "#FFFFFF", "zm": "#0072B2"}


def thumb(ax, img, x, y, size, label=None, below=1.2, fs=6.5):
    ax.imshow(np.asarray(img), extent=(x, x + size, y, y + size), zorder=2, interpolation="lanczos")
    ax.add_patch(Rectangle((x, y), size, size, fill=False, lw=0.5, ec=INK, zorder=3))
    if label:
        ax.text(x + size / 2, y - below, label, ha="center", va="top", fontsize=fs, color=INK, linespacing=1.15)


def box(ax, x, y, w, h, text, fc=FROZEN, ec=FROZEN_EDGE, fs=6.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fc, ec=ec, lw=0.6, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            weight=weight, linespacing=1.2, zorder=3)


def arrow(ax, p, q, rad=0.0, color=INK, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.7, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=4))


def masked(image, mask):
    rgb = np.asarray(image).copy()
    rgb[np.asarray(mask) > 127] = 128   # the masked-image input is zero (mid-gray) inside the edit mask
    return Image.fromarray(rgb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/ocean/projects/cis260330p/shared/data/CelebAMask-HQ")
    ap.add_argument("--eval-run", default="/ocean/projects/cis260330p/shared/runs/wsun8/"
                                          "20261019T163222971419Z_wsun8_evaluate_2745b5d9")
    ap.add_argument("--out", default="results/figures/method_diagram")
    args = ap.parse_args()

    train = load_sample(args.data_root, TRAIN_ID, "hair")
    source = describe_sample(train)["source_color"]
    synthetic = recolor_lab(train["image"], train["mask"], TRAIN_COLOR)
    test = load_sample(args.data_root, TEST_ID, "hair")
    output = Image.open(Path(args.eval_run) / "images" / "Main" / f"{TEST_ID:05d}-hair-{TEST_COLOR}.png").convert("RGB")

    plt.rcParams.update({"font.size": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans"})
    fig = plt.figure(figsize=(7.0, 3.64))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    # (a) synthetic recolor pairs
    ax.text(1, 51.6, "(a) Building synthetic recolor pairs (training data)", fontsize=7.5, weight="bold", va="top")
    thumb(ax, train["image"], 1, 37, 11, "Photo $x$")
    thumb(ax, train["mask"].convert("RGB"), 13.5, 37, 11, "Hair mask $m$")
    box(ax, 27.5, 38, 16, 9, f"Label current color\nmedian CIELAB in $m$\n→ nearest: '{source}'")
    box(ax, 46.5, 38, 16, 9, "New color $c$ ≠ source,\nsampled from\n{red, blue, green,\npurple, pink, brown}")
    box(ax, 65.5, 38, 16, 9, "Recolor inside $m$:\nshift a*b* to $c$,\nkeep L* (shading)")
    thumb(ax, synthetic, 85, 37, 11, f"Target $\\tilde{{x}}$, $c$ = {TRAIN_COLOR}\n\"…with {TRAIN_COLOR} hair\"")
    for x0, x1 in ((24.5, 27.5), (43.5, 46.5), (62.5, 65.5), (81.5, 85)):
        arrow(ax, (x0, 42.5), (x1, 42.5))
    ax.text(54.5, 36.4, f"Each sample: with prob. 0.5 the synthetic pair ($\\tilde{{x}}$, \"{TRAIN_COLOR} hair\"),\n"
                        f"otherwise reconstruction ($x$, \"{source} hair\"). E1: reconstruction only.",
            ha="center", va="top", fontsize=6.2, color=MUTED, linespacing=1.3)
    ax.plot([1, 99], [31.6, 31.6], color="#BBBBBB", lw=0.5)

    # (b) model
    ax.text(1, 30.8, "(b) Model: SD 1.5 inpainting U-Net with LoRA", fontsize=7.5, weight="bold", va="top")
    thumb(ax, masked(test["image"], test["edit_mask"]), 1, 17, 8.5, "Masked photo\n$x \\odot (1-m)$", below=0.6, fs=5.8)
    thumb(ax, test["edit_mask"].convert("RGB"), 1, 2.5, 8.5, None)
    ax.text(10.4, 7.6, "Mask $m$", fontsize=5.8, va="bottom", color=INK)
    box(ax, 12.5, 18.7, 7.5, 5, "VAE\nencoder")
    arrow(ax, (9.5, 21.2), (12.5, 21.2))

    # 9-channel input: [noisy latent | mask | masked-photo latent]
    cx, cw = 24, 5
    for key, y, h, label in (("z", 20.6, 3.4, "$z_t$ (4)"), ("m", 19.6, 1.0, "$m$ (1)"), ("zm", 16.2, 3.4, "$z_m$ (4)")):
        ax.add_patch(Rectangle((cx, y), cw, h, fc=LATENT[key], ec=INK, lw=0.5, zorder=2))
        ax.text(cx + cw / 2, y + h / 2, label, ha="center", va="center", fontsize=5.5, zorder=3,
                color="white" if key == "zm" else INK)
    ax.text(cx + cw / 2, 15.6, "9-ch input\n64×64", ha="center", va="top", fontsize=5.8, color=MUTED)
    ax.plot([16.25, 16.25, 21.2], [18.7, 17.9, 17.9], color=INK, lw=0.7, zorder=4)
    arrow(ax, (21.2, 17.9), (cx, 17.9))
    ax.plot([9.5, 22.4, 22.4], [6.7, 6.7, 20.1], color=INK, lw=0.7, zorder=4)
    arrow(ax, (22.4, 20.1), (cx, 20.1))
    ax.text(cx + cw / 2, 28.6, "$z_t$: noisy target latent\ntrain: VAE($\\tilde{x}$ or $x$) + noise\ninfer: start from noise",
            ha="center", va="top", fontsize=5.6, color=MUTED, linespacing=1.15)

    # U-Net hourglass with LoRA tags
    ux0, ux1, top, bot = 33, 55, 25.5, 13.5
    ax.add_patch(Polygon([(ux0, top), (ux0 + 7, 21.5), (ux1 - 7, 21.5), (ux1, top), (ux1, bot), (ux1 - 7, 17.5),
                          (ux0 + 7, 17.5), (ux0, bot)], closed=True, fc=FROZEN, ec=FROZEN_EDGE, lw=0.6, zorder=2))
    ax.text((ux0 + ux1) / 2, 19.5, "U-Net (frozen)", ha="center", va="center", fontsize=6.5, zorder=3)
    for lx, ly in ((34.2, 22.6), (37.4, 20.0), (48.6, 20.0), (51.8, 22.6), (34.2, 15.0), (51.8, 15.0)):
        ax.add_patch(Rectangle((lx, ly), 2.0, 1.4, fc=LORA, ec=INK, lw=0.4, zorder=4))
    arrow(ax, (cx + cw, 19.5), (ux0, 19.5))

    # text conditioning
    box(ax, 36, 7.0, 16, 3.4, "CLIP text encoder")
    ax.text(44, 4.2, f"\"a photo of a person with {TEST_COLOR} hair\"", ha="center", va="center", fontsize=6,
            style="italic")
    arrow(ax, (44, 5.0), (44, 7.0))
    arrow(ax, (44, 10.4), (44, 17.5))
    ax.text(44.6, 13.6, "cross-attn", fontsize=5.5, color=MUTED, va="center")

    # inference output (top) and training loss (bottom)
    box(ax, 59, 18.7, 7.5, 5, "VAE\ndecoder")
    arrow(ax, (ux1, 21.2), (59, 21.2))
    ax.text(62.75, 26.2, "inference: 30 steps,\nguidance 7.5", ha="center", va="center", fontsize=5.6, color=MUTED,
            linespacing=1.1)
    thumb(ax, output, 69.5, 16.5, 9.5, "Output (raw,\nno paste-back)", below=0.6, fs=5.8)
    arrow(ax, (66.5, 21.2), (69.5, 21.2))
    box(ax, 58, 6.6, 12, 5, "training loss:\nMSE($\\hat{\\epsilon}$, $\\epsilon$) on noise",
        fc="white", ec=FROZEN_EDGE, fs=5.8)
    arrow(ax, (ux1 - 1.5, 14.5), (58, 9.6), color=MUTED, ls="--")

    # legend
    lx = 82
    for i, (fc, ec, label) in enumerate(((LORA, INK, "trained: LoRA, rank 16,\nattention q, k, v, out"),
                                         (FROZEN, FROZEN_EDGE, "frozen SD 1.5"))):
        ax.add_patch(Rectangle((lx, 24.5 - 4 * i), 2.0, 1.4, fc=fc, ec=ec, lw=0.4))
        ax.text(lx + 2.8, 25.2 - 4 * i, label, fontsize=5.8, va="center", linespacing=1.15)
    for i, (key, label) in enumerate((("z", "noisy latent"), ("m", "mask"), ("zm", "masked-photo latent"))):
        ax.add_patch(Rectangle((lx, 16.5 - 3 * i), 2.0, 1.4, fc=LATENT[key], ec=INK, lw=0.4))
        ax.text(lx + 2.8, 17.2 - 3 * i, label, fontsize=5.8, va="center")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
