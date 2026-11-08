import argparse
import json
import statistics as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from color_editing.data import _rgb_to_lab, load_sample

# a band of 3 pixels on each side of the region border
BAND = 7
WHITE = 90


def band(sample):
    m = sample["mask"]
    outer = np.asarray(m.filter(ImageFilter.MaxFilter(BAND))) > 127
    inner = np.asarray(m.filter(ImageFilter.MinFilter(BAND))) > 127
    return outer & ~inner


def near_white(image, keep):
    lab = _rgb_to_lab(np.asarray(image.convert("RGB"))[keep])
    return float((lab[:, 0] > WHITE).mean())


def main(run_dirs, data):
    cases = json.load(open(Path(run_dirs[0]) / "cases.json"))
    samples = {}
    for c in cases:
        key = (c["image_id"], c["region"])
        if key not in samples:
            s = load_sample(data, c["image_id"], c["region"])
            samples[key] = (s["image"], band(s))
    rows = []

    def one(args):
        d, method, c = args
        image, keep = samples[(c["image_id"], c["region"])]
        name = f"{c['image_id']:05d}-{c['region']}-{c['target_color']}.png"
        return {"method": method, "region": c["region"], "target": c["target_color"],
                "orig": near_white(image, keep), "out": near_white(Image.open(d / "images" / method / name), keep)}

    jobs = [(Path(d), m.name, c) for d in run_dirs for m in sorted((Path(d) / "images").iterdir()) for c in cases]
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, jobs))
    print(f"% share of pixels with L* > {WHITE} in a {BAND}px band around the region border, mean over edits")
    groups = {}
    for r in rows:
        base = r["method"].split("-s")[0]
        for t in (r["target"], "all"):
            groups.setdefault((base, r["region"], t), {}).setdefault(r["method"], []).append(r)
    for (base, region, t), by_seed in sorted(groups.items(), key=lambda k: (k[0][1], k[0][2], k[0][0])):
        means = [st.mean(r["out"] for r in v) for v in by_seed.values()]
        orig = st.mean(r["orig"] for v in by_seed.values() for r in v)
        print(base, region, t, len(means), f"orig {100 * orig:.1f}%", f"out {100 * st.mean(means):.1f}%",
              [round(100 * x, 1) for x in means])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--data", type=Path, default=Path("/ocean/projects/cis260330p/shared/data/CelebAMask-HQ"))
    args = parser.parse_args()
    main(args.runs, args.data)
