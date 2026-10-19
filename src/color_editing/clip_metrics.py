import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import COLOR_RGB, load_sample
from .evaluate import case_name
from .pipeline import prompt_for
from .runs import save_json

CLIP_ID = "openai/clip-vit-large-patch14"


def palette(region):
    excluded = "yellow" if region == "hair" else "blond"
    return [c for c in COLOR_RGB if c != excluded]


def region_crop(image, mask, size=224):
    # gray out everything except the region, then crop its bounding box to a square
    rgb = np.asarray(image.convert("RGB")).copy()
    selected = np.asarray(mask) > 127
    rgb[~selected] = 128
    ys, xs = np.nonzero(selected)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) // 2 + 8
    padded = np.pad(rgb, ((half, half), (half, half), (0, 0)), constant_values=128)
    crop = padded[cy:cy + 2 * half, cx:cx + 2 * half]
    return Image.fromarray(crop).resize((size, size), Image.Resampling.BICUBIC)


def features(out):
    import torch
    return out if torch.is_tensor(out) else out.pooler_output


def score_run(eval_run):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    eval_run = Path(eval_run)
    config = json.loads((eval_run / "config.json").read_text())
    cases = json.loads((eval_run / "cases.json").read_text())
    methods = sorted(p.name for p in (eval_run / "images").iterdir() if p.is_dir())
    data_root = Path(config["shared_root"]) / "data/CelebAMask-HQ"
    model = CLIPModel.from_pretrained(CLIP_ID, dtype=dtype).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID)

    def embed_text(texts):
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            return torch.nn.functional.normalize(features(model.get_text_features(**inputs)).float(), dim=-1)

    def embed_images(images):
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = features(model.get_image_features(pixel_values=inputs["pixel_values"].to(dtype)))
        return torch.nn.functional.normalize(feats.float(), dim=-1)

    color_text = {}
    for region in config["regions"]:
        names = palette(region)
        words = [f"a photo of {c} hair" if region == "hair" else f"a photo of {c} clothes" for c in names]
        color_text[region] = (names, embed_text(words))

    samples = {}
    for c in cases:
        key = (c["image_id"], c["region"])
        if key not in samples:
            samples[key] = load_sample(data_root, c["image_id"], c["region"], config["mask_dilation"])

    # label check: does clip agree with the rule-based label on the original photos
    agree = []
    for (image_id, region), s in samples.items():
        names, text = color_text[region]
        sims = embed_images([region_crop(s["image"], s["mask"])]) @ text.T
        source = next(c["source_color"] for c in cases if c["image_id"] == image_id and c["region"] == region)
        agree.append({"image_id": image_id, "region": region, "rule_label": source,
                      "clip_label": names[int(sims.argmax())]})

    rows = []
    for method in methods:
        for start in range(0, len(cases), 32):
            batch = cases[start:start + 32]
            outs = [Image.open(eval_run / "images" / method / f"{case_name(c)}.png").convert("RGB") for c in batch]
            crops = [region_crop(o, samples[(c["image_id"], c["region"])]["mask"]) for o, c in zip(outs, batch)]
            crop_feats = embed_images(crops)
            full_feats = embed_images(outs)
            prompt_feats = embed_text([prompt_for(c["region"], c["target_color"]) for c in batch])
            for i, c in enumerate(batch):
                names, text = color_text[c["region"]]
                predicted = names[int((crop_feats[i] @ text.T).argmax())]
                rows.append({**c, "method": method, "clip_color": predicted,
                             "clip_correct": predicted == c["target_color"],
                             "clip_score": float(100 * (full_feats[i] @ prompt_feats[i]))})

    with (eval_run / "clip_metrics.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    groups = {}
    for r in rows:
        for key in ((r["method"], r["region"], "all"), (r["method"], r["region"], r["target_color"])):
            groups.setdefault(key, []).append(r)
    table = [{"method": m, "region": reg, "target": t, "n": len(items),
              "clip_color_accuracy": float(np.mean([r["clip_correct"] for r in items])),
              "clip_score": float(np.mean([r["clip_score"] for r in items]))}
             for (m, reg, t), items in sorted(groups.items())]
    agreement = {region: float(np.mean([a["rule_label"] == a["clip_label"] for a in agree if a["region"] == region]))
                 for region in config["regions"]}
    save_json(eval_run / "clip_summary.json", {"table": table, "label_agreement": agreement, "labels": agree})
    with (eval_run / "clip_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for row in table:
        if row["target"] == "all":
            print(f"{row['method']:>16} {row['region']:>5} n={row['n']:4d} "
                  f"clip_acc={row['clip_color_accuracy']:.3f} clip_score={row['clip_score']:.2f}", flush=True)
    print("rule vs clip label agreement on originals:", agreement, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", required=True)
    args = parser.parse_args()
    score_run(args.eval_run)
