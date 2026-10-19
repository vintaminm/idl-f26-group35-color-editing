import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageFilter

from .data import COLOR_RGB, _load_mask, _mask_path, describe_sample, recolor_lab, split_ids
from .pipeline import prompt_for
from .runs import require_gpu, save_json


def all_prompts(regions):
    return [prompt_for(region, color) for region in regions for color in COLOR_RGB]


class ImageRegions:
    def __init__(self, data_root, ids, config):
        self.data_root = Path(data_root)
        self.ids = ids
        self.config = config

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        cfg = self.config
        image_id = int(self.ids[index])
        image = None
        items = []
        for region in cfg["regions"]:
            path = _mask_path(self.data_root, image_id, region)
            if not path.is_file():
                continue
            mask = _load_mask(path)
            fraction = float(np.mean(np.asarray(mask) > 127))
            if fraction == 0 or fraction < cfg["min_mask_fraction"]:
                continue
            if image is None:
                with Image.open(self.data_root / "CelebA-HQ-img" / f"{image_id}.jpg") as raw:
                    image = raw.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
            source = describe_sample({"image": image, "mask": mask, "region": region,
                                      "image_id": image_id})["source_color"]
            # synthetic target: a different color, recolored with B0 inside the raw mask
            rng = np.random.default_rng([cfg["seed"], image_id, cfg["regions"].index(region)])
            choices = [c for c in cfg["synthetic_colors"] if c != source]
            target = str(rng.choice(choices))
            recolored = recolor_lab(image, mask, target)
            dilation = cfg["mask_dilation"]
            edit_mask = mask.filter(ImageFilter.MaxFilter(dilation)) if dilation > 1 else mask
            items.append({"region": region, "source": source, "target": target,
                          "fraction": fraction, "edit_mask": np.asarray(edit_mask) > 127,
                          "recolored": np.asarray(recolored)})
        return {"image_id": image_id, "image": None if image is None else np.asarray(image),
                "items": items}


def keep_list(batch):
    return batch


def to_pixels(arrays, device):
    import torch
    x = torch.from_numpy(np.stack(arrays)).to(device)
    return x.permute(0, 3, 1, 2).half() / 127.5 - 1.0


def build_cache(config, split, out_path, limit=None, shard=0, num_shards=1, batch_images=8, workers=4):
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL
    from torch.utils.data import DataLoader
    from transformers import CLIPTextModel, CLIPTokenizer

    device = require_gpu()
    model_dir = Path(config["shared_root"]) / "models/sd15-inpainting"
    data_root = Path(config["shared_root"]) / "data/CelebAMask-HQ"
    ids = split_ids(config["split_seed"])[split]
    if limit:
        ids = ids[:limit]
    ids = ids[shard::num_shards]

    vae = AutoencoderKL.from_pretrained(model_dir, subfolder="vae", variant="fp16",
                                        torch_dtype=torch.float16, local_files_only=True).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(model_dir, subfolder="text_encoder", variant="fp16",
                                                 torch_dtype=torch.float16, local_files_only=True).to(device).eval()
    scale = vae.config.scaling_factor
    prompts = all_prompts(config["regions"])
    prompt_index = {p: i for i, p in enumerate(prompts)}
    generator = torch.Generator(device=device).manual_seed(config["seed"] + 1000 * shard)

    def encode(pixels):
        out = []
        for chunk in pixels.split(16):
            out.append(vae.encode(chunk).latent_dist.sample(generator=generator) * scale)
        return torch.cat(out)

    with torch.inference_mode():
        tokens = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt").input_ids.to(device)
        text_embeds = text_encoder(tokens)[0].cpu()
    del text_encoder

    loader = DataLoader(ImageRegions(data_root, ids, config), batch_size=batch_images,
                        num_workers=workers, collate_fn=keep_list)
    image_ids, image_latents = [], []
    sample_image, masks, masked_latents, synth_latents = [], [], [], []
    records = []
    started = time.perf_counter()
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            batch = [b for b in batch if b["items"]]
            if not batch:
                continue
            pixels = to_pixels([b["image"] for b in batch], device)
            latents = encode(pixels)
            masked, recolored, small = [], [], []
            for i, b in enumerate(batch):
                for item in b["items"]:
                    m = torch.from_numpy(item["edit_mask"]).to(device)[None].half()
                    masked.append(pixels[i] * (1 - m))
                    recolored.append(item["recolored"])
                    small.append(F.interpolate(m[None].float(), size=latents.shape[-2:], mode="nearest")[0] > 0.5)
                    sample_image.append(len(image_ids) + i)
                    records.append({"image_id": b["image_id"], "region": item["region"],
                                    "source_color": item["source"], "synthetic_color": item["target"],
                                    "mask_fraction": item["fraction"],
                                    "source_prompt": prompt_index[prompt_for(item["region"], item["source"])],
                                    "synthetic_prompt": prompt_index[prompt_for(item["region"], item["target"])]})
            masked_lat = encode(torch.stack(masked))
            synth_lat = encode(to_pixels(recolored, device))
            for t in (latents, masked_lat, synth_lat):
                if not torch.isfinite(t).all():
                    raise RuntimeError("non-finite latents from the vae")
            image_ids += [b["image_id"] for b in batch]
            image_latents.append(latents.cpu())
            masked_latents.append(masked_lat.cpu())
            synth_latents.append(synth_lat.cpu())
            masks.append(torch.stack(small).cpu())
            if step % 50 == 0:
                rate = len(image_ids) / (time.perf_counter() - started)
                print(f"{len(image_ids)}/{len(ids)} images, {len(records)} region samples, {rate:.1f} images/s",
                      flush=True)

    cache = {
        "image_ids": torch.tensor(image_ids),
        "image_latents": torch.cat(image_latents),
        "sample_image": torch.tensor(sample_image),
        "masks": torch.cat(masks),
        "masked_latents": torch.cat(masked_latents),
        "synthetic_latents": torch.cat(synth_latents),
        "source_prompt": torch.tensor([r["source_prompt"] for r in records]),
        "synthetic_prompt": torch.tensor([r["synthetic_prompt"] for r in records]),
        "text_embeds": text_embeds,
        "prompts": prompts,
        "records": records,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    summary = {"split": split, "shard": shard, "num_shards": num_shards, "limit": limit,
               "images": len(image_ids), "region_samples": len(records),
               "regions": {r: sum(x["region"] == r for x in records) for r in config["regions"]},
               "seconds": time.perf_counter() - started, "config": config}
    save_json(out_path.with_suffix(".json"), summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}), flush=True)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    build_cache(json.loads(Path(args.config).read_text()), args.split, args.out, args.limit,
                args.shard, args.num_shards, workers=args.workers)
