"""FID and KID realism of edited images against unedited test photos (T2).

Reads a finished evaluation run and writes a new run under results/ with
Inception-v3 (2048-d, the torch-fidelity "inception-v3-compat" extractor)
distances between each method's outputs and real CelebA-HQ test photos that
were never edited. Two views are scored:

- full: the saved 512x512 output against whole reference photos.
- region: a square crop around the edited region, everything else greyed
  out (the same crop as the CLIP metric), against the same crop of real
  hair or clothing regions.

Also scored, as floors: the originals of the edited photos ("original") and
their round trip through the frozen SD 1.5 VAE with no edit ("vae").

FID is biased upward for small samples (under ~1k images), so KID, which is
unbiased and reported with a standard deviation over random subsets, is the
number to compare methods with; per-target rows (~100 images) carry KID only
in spirit, FID is still written but should not be read closely.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .clip_metrics import region_crop
from .data import _mask_path, load_sample, split_ids
from .evaluate import case_name
from .runs import create_run, finish_run, repo_root, save_json

CROP = 299
KID_SUBSETS = 100
KID_SUBSET_SIZE = 100


def load_image(data_root, image_id):
    """The photo as load_sample reads it, without needing a region mask."""
    with Image.open(Path(data_root) / "CelebA-HQ-img" / f"{image_id}.jpg") as raw:
        return raw.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)


def to_uint8(images):
    import torch
    arr = np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in images])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


class Extractor:
    def __init__(self, device):
        import torch
        os.environ.setdefault("TORCH_HOME", str(repo_root() / "results" / "_torch_cache"))
        from torchmetrics.image.fid import NoTrainInceptionV3
        self.device = device
        self.net = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
        self.batch = 64 if device.type == "cuda" else 8

    def __call__(self, images):
        import torch
        feats = []
        for start in range(0, len(images), self.batch):
            x = to_uint8(images[start:start + self.batch]).to(self.device)
            with torch.no_grad():
                feats.append(self.net(x).double().cpu().numpy())
        return np.concatenate(feats) if feats else np.zeros((0, 2048))


def fid(x, y, device=None):
    """Frechet distance between Gaussians fitted to two feature sets.

    tr(sqrtm(S1 S2)) is the sum of square roots of the eigenvalues of S1 S2,
    which are real and non-negative for covariance matrices; this avoids the
    slow dense matrix square root and runs on the GPU when one is present.
    """
    import torch
    mu1, mu2 = x.mean(0), y.mean(0)
    s1, s2 = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    prod = torch.from_numpy(s1 @ s2).to(device or "cpu")
    eig = torch.linalg.eigvals(prod).real.clamp(min=0).sqrt().sum().item()
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(s1) + np.trace(s2) - 2 * eig)


def kid(x, y, subset_size, subsets, seed):
    """Unbiased MMD^2 with the cubic polynomial kernel, averaged over random subsets."""
    rng = np.random.default_rng(seed)
    m = min(subset_size, len(x), len(y))
    d = x.shape[1]
    values = []
    for _ in range(subsets):
        a = x[rng.choice(len(x), m, replace=False)]
        b = y[rng.choice(len(y), m, replace=False)]
        kaa = (a @ a.T / d + 1) ** 3
        kbb = (b @ b.T / d + 1) ** 3
        kab = (a @ b.T / d + 1) ** 3
        mmd = ((kaa.sum() - np.trace(kaa)) / (m * (m - 1)) + (kbb.sum() - np.trace(kbb)) / (m * (m - 1))
               - 2 * kab.mean())
        values.append(mmd)
    return float(np.mean(values)), float(np.std(values)), m


def vae_roundtrip(images, model_dir, device):
    """Encode and decode each image through the frozen SD 1.5 VAE with no edit."""
    import torch
    from diffusers import AutoencoderKL
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(str(model_dir), subfolder="vae", variant="fp16", torch_dtype=dtype,
                                        use_safetensors=True, local_files_only=True).to(device).eval()
    out = []
    for im in images:
        x = torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1)[None] / 127.5 - 1.0
        with torch.no_grad():
            latents = vae.encode(x.to(device, dtype)).latent_dist.mode()
            y = vae.decode(latents).sample
        y = ((y[0].float().clamp(-1, 1) + 1) * 127.5).round().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        out.append(Image.fromarray(y))
    del vae
    return out


def score_run(eval_run, limit=None, ref_limit=None, methods=None, vae=True,
              kid_subsets=KID_SUBSETS, kid_subset_size=KID_SUBSET_SIZE):
    import torch
    eval_run = Path(eval_run).resolve(strict=True)
    config = json.loads((eval_run / "config.json").read_text())
    all_cases = json.loads((eval_run / "cases.json").read_text())
    cases = all_cases[:limit] if limit else all_cases
    found = sorted(p.name for p in (eval_run / "images").iterdir() if p.is_dir())
    methods = [m for m in found if not methods or m in methods]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(8, len(os.sched_getaffinity(0)))))
    run = create_run("fid", {"eval_run": str(eval_run), "shared_root": config["shared_root"],
                             "mask_dilation": config["mask_dilation"], "split_seed": config["split_seed"],
                             "seed": config["seed"], "crop": CROP, "kid_subsets": kid_subsets,
                             "kid_subset_size": kid_subset_size, "device": device.type,
                             "methods": methods, "limit": limit, "ref_limit": ref_limit, "vae": vae})
    print(f"{len(cases)} cases, methods {methods}, device {device}", flush=True)
    try:
        table = _score(eval_run, config, all_cases, cases, methods, run, device, ref_limit, vae,
                       kid_subsets, kid_subset_size)
        finish_run(run, cases=len(cases), methods=methods)
    except BaseException as error:
        finish_run(run, status="failed", error=f"{type(error).__name__}: {error}")
        raise
    print(f"finished {run}", flush=True)
    return run, table


def _score(eval_run, config, all_cases, cases, methods, run, device, ref_limit, vae, kid_subsets, kid_subset_size):
    data_root = Path(config["shared_root"]) / "data/CelebAMask-HQ"
    extract = Extractor(device)
    regions = config["regions"]
    feats = {}  # (row name, region, target, view) -> features

    # reference: test-split photos that were never edited (excluded by image id across all cases)
    edited = {c["image_id"] for c in all_cases}
    ref_ids = [i for i in split_ids(config["split_seed"])["test"] if i not in edited]
    if ref_limit:
        ref_ids = ref_ids[:ref_limit]
    started = time.perf_counter()
    feats[("reference", "all", "all", "full")] = extract([load_image(data_root, i) for i in ref_ids])
    for region in regions:
        crops = []
        for i in ref_ids:
            if _mask_path(data_root, i, region).is_file():
                s = load_sample(data_root, i, region, config["mask_dilation"])
                if np.any(np.asarray(s["mask"]) > 127):
                    crops.append(region_crop(s["image"], s["mask"], CROP))
        feats[("reference", region, "all", "region")] = extract(crops)
    print(f"reference: {len(ref_ids)} photos, "
          + ", ".join(f"{r} crops {len(feats[('reference', r, 'all', 'region')])}" for r in regions)
          + f", {time.perf_counter() - started:.0f}s", flush=True)

    # originals of the edited photos, per region, and their VAE round trip
    samples = {}
    for c in cases:
        key = (c["image_id"], c["region"])
        if key not in samples:
            samples[key] = load_sample(data_root, c["image_id"], c["region"], config["mask_dilation"])
    for region in regions:
        keys = [k for k in samples if k[1] == region]
        if not keys:
            continue
        originals = [samples[k]["image"] for k in keys]
        feats[("original", region, "all", "full")] = extract(originals)
        feats[("original", region, "all", "region")] = extract(
            [region_crop(samples[k]["image"], samples[k]["mask"], CROP) for k in keys])
        if vae:
            started = time.perf_counter()
            decoded = vae_roundtrip(originals, Path(config["shared_root"]) / "models/sd15-inpainting", device)
            feats[("vae", region, "all", "full")] = extract(decoded)
            feats[("vae", region, "all", "region")] = extract(
                [region_crop(d, samples[k]["mask"], CROP) for d, k in zip(decoded, keys)])
            print(f"vae round trip {region}: {len(keys)} photos, {time.perf_counter() - started:.0f}s", flush=True)

    # each method: full outputs and region crops, grouped by region and by target
    for method in methods:
        started = time.perf_counter()
        groups = {}
        for c in cases:
            out = Image.open(eval_run / "images" / method / f"{case_name(c)}.png").convert("RGB")
            crop = region_crop(out, samples[(c["image_id"], c["region"])]["mask"], CROP)
            groups.setdefault((c["region"], "all"), ([], []))
            groups.setdefault((c["region"], c["target_color"]), ([], []))
            for key in ((c["region"], "all"), (c["region"], c["target_color"])):
                groups[key][0].append(out)
                groups[key][1].append(crop)
        for (region, target), (fulls, crops) in groups.items():
            feats[(method, region, target, "full")] = extract(fulls)
            feats[(method, region, target, "region")] = extract(crops)
        print(f"{method}: {len(cases)} edits, {time.perf_counter() - started:.0f}s", flush=True)

    # keep features so seeds can be pooled later without re-extracting
    from safetensors.numpy import save_file
    save_file({"|".join(k): v.astype(np.float32) for k, v in feats.items()}, str(run / "features.safetensors"))

    table = []
    ref_full = feats[("reference", "all", "all", "full")]
    started = time.perf_counter()
    for (name, region, target, view), x in sorted(feats.items()):
        if name == "reference" or view != "full":
            continue
        crop = feats[(name, region, target, "region")]
        ref_region = feats[("reference", region, "all", "region")]
        kf, kf_std, mf = kid(ref_full, x, kid_subset_size, kid_subsets, config["seed"])
        kr, kr_std, mr = kid(ref_region, crop, kid_subset_size, kid_subsets, config["seed"])
        # FID only where the sample is the whole region (~290 images); per-target FID on ~100 is not meaningful
        whole = target == "all"
        table.append({"method": name, "region": region, "target": target, "n": len(x),
                      "n_ref_full": len(ref_full), "fid_full": fid(ref_full, x, device) if whole else None,
                      "kid_full": kf, "kid_full_std": kf_std,
                      "n_ref_region": len(ref_region), "fid_region": fid(ref_region, crop, device) if whole else None,
                      "kid_region": kr, "kid_region_std": kr_std, "kid_subset": min(mf, mr)})
    print(f"distances: {len(table)} rows, {time.perf_counter() - started:.0f}s", flush=True)
    save_json(run / "fid_summary.json", table)
    with (run / "fid_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for row in table:
        if row["target"] == "all":
            print(f"{row['method']:>16} {row['region']:>5} n={row['n']:4d} "
                  f"fid_full={row['fid_full']:7.2f} kid_full={1000 * row['kid_full']:6.2f}e-3 "
                  f"fid_region={row['fid_region']:7.2f} kid_region={1000 * row['kid_region']:6.2f}e-3", flush=True)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-run", required=True, help="finished evaluation run with cases.json and images/")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N cases (smoke test)")
    parser.add_argument("--ref-limit", type=int, default=None, help="use only the first N reference photos")
    parser.add_argument("--methods", nargs="*", default=None, help="subset of image folders to score")
    parser.add_argument("--no-vae", action="store_true", help="skip the VAE round-trip floor")
    parser.add_argument("--kid-subsets", type=int, default=KID_SUBSETS)
    parser.add_argument("--kid-subset-size", type=int, default=KID_SUBSET_SIZE)
    args = parser.parse_args()
    score_run(args.eval_run, limit=args.limit, ref_limit=args.ref_limit, methods=args.methods,
              vae=not args.no_vae, kid_subsets=args.kid_subsets, kid_subset_size=args.kid_subset_size)
