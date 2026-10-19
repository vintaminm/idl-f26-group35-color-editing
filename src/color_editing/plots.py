import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def smooth(values, weight=0.9):
    out, last = [], None
    for v in values:
        last = v if last is None else weight * last + (1 - weight) * v
        out.append(last)
    return out


def curves(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    styles = {"recon": "-", "synthetic": "--"}
    for (label, run), color in zip(runs, ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
        train = read_jsonl(Path(run) / "loss.jsonl")
        val = read_jsonl(Path(run) / "val.jsonl")
        steps = [r["step"] for r in train]
        axes[0].plot(steps, smooth([r["loss"] for r in train]), color=color, label=label)
        for kind, style in styles.items():
            axes[1].plot([r["step"] for r in val], [r[kind] for r in val], style, color=color,
                         marker="o", markersize=3, label=f"{label}, {kind} pairs")
    axes[0].set_title("training loss (smoothed)")
    axes[1].set_title("validation loss, fixed noise and timesteps")
    for ax in axes:
        ax.set_xlabel("step")
        ax.set_ylabel("noise-prediction MSE")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print("saved", out)


def grid(eval_run, out, methods, per_region=2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eval_run = Path(eval_run)
    cases = json.loads((eval_run / "cases.json").read_text())
    picked = []
    for region in ("hair", "cloth"):
        ids = []
        for c in cases:
            if c["region"] == region and c["image_id"] not in ids:
                ids.append(c["image_id"])
        ids = ids[:per_region]
        picked += [c for c in cases if c["region"] == region and c["image_id"] in ids][:per_region * 2]
    columns = ["original", *methods]
    fig, axes = plt.subplots(len(picked), len(columns), figsize=(1.6 * len(columns), 1.7 * len(picked)))
    data_root = Path(json.loads((eval_run / "config.json").read_text())["shared_root"]) / "data/CelebAMask-HQ"
    for i, c in enumerate(picked):
        name = f"{c['image_id']:05d}-{c['region']}-{c['target_color']}"
        with Image.open(data_root / "CelebA-HQ-img" / f"{c['image_id']}.jpg") as raw:
            images = [raw.convert("RGB").resize((256, 256))]
        for m in methods:
            with Image.open(eval_run / "images" / m / f"{name}.png") as im:
                images.append(im.convert("RGB").resize((256, 256)))
        for j, img in enumerate(images):
            ax = axes[i, j]
            ax.imshow(np.asarray(img))
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(columns[j], fontsize=9)
        axes[i, 0].set_ylabel(f"{c['region']}: {c['source_color']}\n-> {c['target_color']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=[], help="label=path/to/training/run")
    parser.add_argument("--eval-run", default=None)
    parser.add_argument("--methods", nargs="*", default=["B0", "B1"])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.runs:
        curves([tuple(r.split("=", 1)) for r in args.runs], out_dir / "loss_curves.png")
    if args.eval_run:
        grid(args.eval_run, out_dir / "qualitative.png", args.methods)
