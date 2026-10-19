import argparse
import json
from pathlib import Path
import time

from .pipeline import LORA_TARGETS, _save_adapter
from .runs import append_jsonl, create_run, finish_run, require_gpu, save_json


def load_cache(paths, device, fraction=1.0):
    import torch

    parts = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    for part in parts[1:]:
        if part["prompts"] != parts[0]["prompts"]:
            raise ValueError("cache shards were built with different prompt lists")
    offsets, total = [], 0
    for part in parts:
        offsets.append(total)
        total += len(part["image_ids"])
    cache = {
        "image_latents": torch.cat([p["image_latents"] for p in parts]),
        "sample_image": torch.cat([p["sample_image"] + off for p, off in zip(parts, offsets)]),
        "text_embeds": parts[0]["text_embeds"],
    }
    for key in ("masks", "masked_latents", "synthetic_latents", "source_prompt", "synthetic_prompt"):
        cache[key] = torch.cat([p[key] for p in parts])
    records = [r for p in parts for r in p["records"]]
    if fraction < 1:
        # cached images are in random order, so the first ones are a random subset
        keep = cache["sample_image"] < round(fraction * len(cache["image_latents"]))
        for key in ("sample_image", "masks", "masked_latents", "synthetic_latents", "source_prompt", "synthetic_prompt"):
            cache[key] = cache[key][keep]
        records = [r for r, k in zip(records, keep.tolist()) if k]
    cache = {k: v.to(device) for k, v in cache.items()}
    cache["records"] = records
    return cache


def make_batch(cache, idx, synthetic):
    import torch

    recon = cache["image_latents"][cache["sample_image"][idx]]
    target = torch.where(synthetic[:, None, None, None], cache["synthetic_latents"][idx], recon)
    prompt = torch.where(synthetic, cache["synthetic_prompt"][idx], cache["source_prompt"][idx])
    mask = cache["masks"][idx].half()
    return target, mask, cache["masked_latents"][idx], cache["text_embeds"][prompt]


def noise_loss(unet, scheduler, target, mask, masked, text, noise, t):
    import torch
    import torch.nn.functional as F

    # 9-channel input: noisy latent, mask, masked-image latent
    noisy = scheduler.add_noise(target.float(), noise, t)
    x = torch.cat([noisy.half(), mask, masked], dim=1)
    with torch.autocast("cuda", dtype=torch.float16):
        pred = unet(x, t, encoder_hidden_states=text).sample
    return F.mse_loss(pred.float(), noise, reduction="none").mean(dim=(1, 2, 3))


def validation_loss(unet, scheduler, val, fixed, batch_size=16):
    import torch

    unet.eval()
    out = {}
    with torch.no_grad():
        for kind, synthetic in (("recon", False), ("synthetic", True)):
            losses = []
            for start in range(0, len(fixed["idx"]), batch_size):
                idx = fixed["idx"][start:start + batch_size]
                flag = torch.full((len(idx),), synthetic, device=idx.device)
                target, mask, masked, text = make_batch(val, idx, flag)
                losses.append(noise_loss(unet, scheduler, target, mask, masked, text,
                                         fixed["noise"][start:start + batch_size],
                                         fixed["t"][start:start + batch_size]))
            out[kind] = float(torch.cat(losses).mean())
    unet.train()
    return out


def train(config, train_paths, val_paths):
    import torch
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig

    device = require_gpu()
    tr = config["training"]
    model_dir = Path(config["shared_root"]) / "models/sd15-inpainting"
    torch.manual_seed(config["seed"])
    gen = torch.Generator(device=device).manual_seed(config["seed"])

    data = load_cache(train_paths, device, tr.get("data_fraction", 1.0))
    val = load_cache(val_paths, device)
    n_val = min(tr["val_samples"], len(val["records"]))
    val_gen = torch.Generator(device=device).manual_seed(config["split_seed"] + 1)
    # fixed noise and timesteps so the val curve only moves with the model
    fixed = {
        "idx": torch.randperm(len(val["records"]), generator=val_gen, device=device)[:n_val],
        "noise": torch.randn((n_val, 4, 64, 64), generator=val_gen, device=device),
        "t": (torch.arange(n_val, device=device) * 1000 // n_val)[
            torch.randperm(n_val, generator=val_gen, device=device)],
    }

    unet = UNet2DConditionModel.from_pretrained(model_dir, subfolder="unet", variant="fp16",
                                                torch_dtype=torch.float16, local_files_only=True)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=tr["rank"], lora_alpha=tr["rank"], init_lora_weights="gaussian",
                                target_modules=LORA_TARGETS))
    params = []
    for name, p in unet.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
            params.append(p)
    unet.to(device).train()
    scheduler = DDPMScheduler.from_pretrained(model_dir, subfolder="scheduler", local_files_only=True)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    optimizer = torch.optim.AdamW(params, lr=tr["learning_rate"], weight_decay=tr.get("weight_decay", 0.01))
    scaler = torch.amp.GradScaler("cuda")

    run = create_run(config["name"], {**config, "train_cache": [str(p) for p in train_paths],
                                      "val_cache": [str(p) for p in val_paths]})
    n_params = sum(p.numel() for p in params)
    save_json(run / "data.json", {"train_region_samples": len(data["records"]),
                                  "train_images": len(data["image_latents"]),
                                  "val_samples_used": n_val, "lora_parameters": n_params})
    print(f"run {run.name}: {len(data['records'])} train samples, {n_params:,} lora parameters", flush=True)

    def log_val(step):
        v = validation_loss(unet, scheduler, val, fixed)
        append_jsonl(run / "val.jsonl", {"step": step, **v})
        print(f"step {step} val recon {v['recon']:.5f} synthetic {v['synthetic']:.5f}", flush=True)

    try:
        log_val(0)
        bs = tr["batch_size"]
        sums = {"recon": [0.0, 0], "synthetic": [0.0, 0]}
        started = time.perf_counter()
        for step in range(1, tr["steps"] + 1):
            idx = torch.randint(0, len(data["records"]), (bs,), generator=gen, device=device)
            synthetic = torch.rand(bs, generator=gen, device=device) < tr["synthetic_fraction"]
            target, mask, masked, text = make_batch(data, idx, synthetic)
            noise = torch.randn(target.shape, generator=gen, device=device)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (bs,), generator=gen, device=device)
            losses = noise_loss(unet, scheduler, target, mask, masked, text, noise, t)
            loss = losses.mean()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            for kind, flag in (("recon", ~synthetic), ("synthetic", synthetic)):
                if flag.any():
                    sums[kind][0] += float(losses[flag].sum())
                    sums[kind][1] += int(flag.sum())
            if step % tr["log_every"] == 0:
                row = {"step": step, "grad_norm": float(grad_norm),
                       "elapsed_seconds": time.perf_counter() - started}
                for kind, (total, count) in sums.items():
                    row[kind] = total / count if count else None
                total = sum(v[0] for v in sums.values())
                row["loss"] = total / sum(v[1] for v in sums.values())
                append_jsonl(run / "loss.jsonl", row)
                sums = {"recon": [0.0, 0], "synthetic": [0.0, 0]}
            if step % 500 == 0:
                print(f"step {step}/{tr['steps']} loss {row['loss']:.5f} "
                      f"{(time.perf_counter() - started) / step:.3f}s/step", flush=True)
            if step % tr["val_every"] == 0:
                log_val(step)
            if step % tr["save_every"] == 0 and step < tr["steps"]:
                _save_adapter(unet, run / "checkpoints" / f"step-{step:06d}", step, config)
        _save_adapter(unet, run / "adapter", tr["steps"], config)
        finish_run(run, steps=tr["steps"], lora_parameters=n_params,
                   peak_cuda_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                   seconds=time.perf_counter() - started, gpu=torch.cuda.get_device_name(0))
    except BaseException as error:
        finish_run(run, status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   error=f"{type(error).__name__}: {error}")
        raise
    print(f"finished {run}", flush=True)
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-cache", nargs="+", required=True)
    parser.add_argument("--val-cache", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.steps:
        cfg["training"]["steps"] = args.steps
    if args.seed is not None:
        cfg["seed"] = args.seed
        cfg["name"] = f"{cfg['name']}-seed{args.seed}"
    train(cfg, [Path(p) for p in args.train_cache], [Path(p) for p in args.val_cache])
