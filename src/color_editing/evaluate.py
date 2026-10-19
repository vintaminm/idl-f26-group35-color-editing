import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .data import COLOR_RGB, _load_mask, _mask_path, _rgb_to_lab, describe_sample, load_sample, recolor_lab, split_ids
from .pipeline import prompt_for
from .runs import append_jsonl, create_run, finish_run, require_gpu, save_json


def test_cases(config):
    data_root = Path(config["shared_root"]) / "data/CelebAMask-HQ"
    ev = config["evaluation"]
    ids = split_ids(config["split_seed"])["test"]
    samples, cases = {}, []
    for region in config["regions"]:
        count = 0
        for image_id in ids:
            path = _mask_path(data_root, image_id, region)
            if not path.is_file():
                continue
            fraction = float(np.mean(np.asarray(_load_mask(path)) > 127))
            if fraction == 0 or fraction < config["min_mask_fraction"]:
                continue
            sample = load_sample(data_root, image_id, region, config["mask_dilation"])
            source = describe_sample(sample)["source_color"]
            samples[(image_id, region)] = sample
            for target in ev["targets"][region]:
                if target != source:
                    cases.append({"image_id": image_id, "region": region, "source_color": source,
                                  "target_color": target, "prompt": prompt_for(region, target),
                                  "seed": config["seed"] + image_id})
            count += 1
            if count == ev["cases_per_region"]:
                break
    return samples, cases


def case_name(case):
    return f"{case['image_id']:05d}-{case['region']}-{case['target_color']}"


def generate(config, samples, cases, run, methods):
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    device = require_gpu()
    model_dir = Path(config["shared_root"]) / "models/sd15-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        str(model_dir), variant="fp16", torch_dtype=torch.float16, use_safetensors=True,
        local_files_only=True, safety_checker=None, requires_safety_checker=False).to(device)
    pipe.set_progress_bar_config(disable=True)
    inf = config["inference"]
    bs = config["evaluation"].get("batch_size", 4)
    for method, adapter in methods:
        if adapter:
            pipe.load_lora_weights(str(adapter), local_files_only=True)
        out_dir = run / "images" / method
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        for start in range(0, len(cases), bs):
            batch = cases[start:start + bs]
            picked = [samples[(c["image_id"], c["region"])] for c in batch]
            with torch.inference_mode():
                images = pipe(prompt=[c["prompt"] for c in batch],
                              image=[s["image"] for s in picked],
                              mask_image=[s["edit_mask"] for s in picked],
                              height=512, width=512, strength=1.0,
                              num_inference_steps=inf["steps"], guidance_scale=inf["guidance_scale"],
                              generator=[torch.Generator(device=device).manual_seed(c["seed"]) for c in batch],
                              ).images
            for c, image in zip(batch, images):
                image.convert("RGB").save(out_dir / f"{case_name(c)}.png")
            if start % (bs * 25) == 0:
                print(f"{method}: {start + len(batch)}/{len(cases)} "
                      f"{(time.perf_counter() - started) / (start + len(batch)):.2f}s/image", flush=True)
        if adapter:
            pipe.unload_lora_weights()
    del pipe
    torch.cuda.empty_cache()


def score(config, samples, cases, run, method_names):
    import lpips
    import torch

    device = require_gpu()
    metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def tensor(img):
        x = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1)[None]
        return (x / 127.5 - 1.0).to(device)

    rows = []
    for method in ["B0", *method_names]:
        for c in cases:
            sample = samples[(c["image_id"], c["region"])]
            if method == "B0":
                out = recolor_lab(sample["image"], sample["mask"], c["target_color"])
                out.save(run / "images" / "B0" / f"{case_name(c)}.png")
            else:
                out = Image.open(run / "images" / method / f"{case_name(c)}.png").convert("RGB")
            measured = describe_sample({**sample, "image": out})
            target_lab = _rgb_to_lab(np.asarray(COLOR_RGB[c["target_color"]]))
            # changes outside the edit mask: put the original back inside it, then compare
            restored = Image.composite(sample["image"], out, sample["edit_mask"])
            outside = np.asarray(sample["edit_mask"]) == 0
            diff = np.abs(np.asarray(out, dtype=np.float32) - np.asarray(sample["image"], dtype=np.float32))
            with torch.no_grad():
                lp = float(metric(tensor(restored), tensor(sample["image"])))
            rows.append({**c, "method": method, "predicted_color": measured["source_color"],
                         "correct": measured["source_color"] == c["target_color"],
                         "delta_e": float(np.linalg.norm(np.asarray(measured["median_lab"]) - target_lab)),
                         "lpips_outside": lp, "mae_outside": float(diff[outside].mean() / 255.0)})
            append_jsonl(run / "metrics.jsonl", rows[-1])
    return rows


def summarize(rows, run):
    groups = {}
    for r in rows:
        for key in ((r["method"], r["region"], "all"), (r["method"], r["region"], r["target_color"])):
            groups.setdefault(key, []).append(r)
    table = []
    for (method, region, target), items in sorted(groups.items()):
        table.append({"method": method, "region": region, "target": target, "n": len(items),
                      "color_accuracy": float(np.mean([r["correct"] for r in items])),
                      "delta_e": float(np.mean([r["delta_e"] for r in items])),
                      "lpips_outside": float(np.mean([r["lpips_outside"] for r in items])),
                      "mae_outside": float(np.mean([r["mae_outside"] for r in items]))})
    save_json(run / "summary.json", table)
    with (run / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for row in table:
        if row["target"] == "all":
            print(f"{row['method']:>6} {row['region']:>5} n={row['n']:4d} acc={row['color_accuracy']:.3f} "
                  f"dE={row['delta_e']:6.2f} lpips_out={row['lpips_outside']:.4f} "
                  f"mae_out={row['mae_outside']:.4f}", flush=True)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", nargs="*", default=[], help="name=path/to/adapter")
    parser.add_argument("--cases-per-region", type=int, default=None)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.cases_per_region:
        cfg["evaluation"]["cases_per_region"] = args.cases_per_region
    methods = [("B1", None)] + [tuple(a.split("=", 1)) for a in args.adapter]
    run = create_run("evaluate", {**cfg, "adapters": dict(methods)})
    try:
        samples, cases = test_cases(cfg)
        save_json(run / "cases.json", cases)
        print(f"{len(cases)} cases from {len(samples)} test regions", flush=True)
        (run / "images" / "B0").mkdir(parents=True)
        generate(cfg, samples, cases, run, methods)
        rows = score(cfg, samples, cases, run, [m for m, _ in methods])
        summarize(rows, run)
        finish_run(run, cases=len(cases), methods=["B0"] + [m for m, _ in methods])
    except BaseException as error:
        finish_run(run, status="failed", error=f"{type(error).__name__}: {error}")
        raise
    print(f"finished {run}", flush=True)
