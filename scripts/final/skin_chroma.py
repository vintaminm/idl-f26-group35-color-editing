import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from color_editing.data import _load_mask, _rgb_to_lab, load_sample

# skip a 10 px band around the edit mask so boundary blending does not count
MARGIN = 21
MIN_PIXELS = 1000


def skin_pixels(sample, skin_root):
    i = sample["image_id"]
    skin = np.asarray(_load_mask(skin_root / str(i // 2000) / f"{i:05d}_skin.png")) > 0
    near = np.asarray(sample["mask"].filter(ImageFilter.MaxFilter(MARGIN))) > 0
    return skin & ~near


def chroma_lightness(image, keep):
    lab = _rgb_to_lab(np.asarray(image.convert("RGB"))[keep])
    return float(np.hypot(lab[:, 1], lab[:, 2]).mean()), float(lab[:, 0].mean())


def main(run_dirs, data, skin_root, rows_out):
    originals = {}
    rows = []
    for d in map(Path, run_dirs):
        cases = json.load(open(d / "cases.json"))
        for method in sorted(p.name for p in (d / "images").iterdir()):
            for c in cases:
                key = (c["image_id"], c["region"])
                if key not in originals:
                    s = load_sample(data, c["image_id"], c["region"])
                    keep = skin_pixels(s, skin_root)
                    originals[key] = (keep, *chroma_lightness(s["image"], keep)) if keep.sum() >= MIN_PIXELS else None
                if originals[key] is None:
                    continue
                keep, c0, l0 = originals[key]
                name = f"{c['image_id']:05d}-{c['region']}-{c['target_color']}"
                c1, l1 = chroma_lightness(Image.open(d / "images" / method / f"{name}.png"), keep)
                rows.append({"run": d.name[-8:], "method": method, "region": c["region"], "target": c["target_color"],
                             "c0": c0, "dc": c1 - c0, "dl": l1 - l0})
    skipped = sum(v is None for v in originals.values())
    print(f"% skin pixels outside a {MARGIN}px max-filtered edit mask; {skipped} image-region pairs skipped (< {MIN_PIXELS} px)")
    print("run method region target n C*orig dC* (mean, sd) dL* share(dC*<-2)")
    groups = {}
    for r in rows:
        for t in (r["target"], "all"):
            groups.setdefault((r["run"], r["method"], r["region"], t), []).append(r)
    for (run, m, region, t), g in groups.items():
        dc = [r["dc"] for r in g]
        print(run, m, region, t, len(g), f"{st.mean(r['c0'] for r in g):.2f}", f"{st.mean(dc):+.2f}",
              f"{st.stdev(dc):.2f}", f"{st.mean(r['dl'] for r in g):+.2f}", f"{sum(x < -2 for x in dc) / len(dc):.2f}")
    json.dump(rows, open(rows_out, "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--data", type=Path, default=Path("/ocean/projects/cis260330p/shared/data/CelebAMask-HQ"))
    # skin masks; defaults to the mask folder of --data
    parser.add_argument("--skin", type=Path)
    parser.add_argument("--rows", default="skin_chroma_rows.json")
    args = parser.parse_args()
    main(args.runs, args.data, args.skin or args.data / "CelebAMask-HQ-mask-anno", args.rows)
