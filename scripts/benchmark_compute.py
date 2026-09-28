"""Measure the compute cost of the project plan on the current device.

Loads the real Stable Diffusion v1.5 inpainting U-Net and VAE, adds LoRA to the
attention projections, and measures
  - trainable LoRA parameters for each rank,
  - FLOPs per training sample and per generated image (torch FlopCounterMode),
  - training throughput (samples/s) and generation throughput (images/s),
  - peak memory (CUDA only),
then turns them into device-hours for configs/compute_plan.json.

Training uses precomputed latents and text embeddings, as in the plan, so the VAE
and text encoder are not in the training loop. Inputs are random tensors with the
real shapes; speed does not depend on pixel values.

Example:
  python scripts/benchmark_compute.py --batch-size 8 --train-steps 20
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]  # attention projections
TEXT_LEN, TEXT_DIM = 77, 768  # CLIP ViT-L/14 text embedding shape


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_unet(device, rank):
    # Frozen fp16 base weights; LoRA weights A, B kept in fp32 for the optimizer.
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", variant="fp16", torch_dtype=torch.float16
    )
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                target_modules=LORA_TARGETS))
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if ARGS.grad_checkpointing:
        unet.enable_gradient_checkpointing()
    return unet.to(device)


def lora_stats(unet):
    n_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    n_layers = sum(1 for n, _ in unet.named_modules() if n.endswith("lora_A.default"))
    return n_params, n_layers


def make_batch(bs, latent, device):
    # U-Net input: noisy latent z_t (4) + mask m (1) + masked-image latent E(x*(1-m)) (4) = 9 channels.
    z0 = torch.randn(bs, 4, latent, latent, device=device)
    mask = (torch.rand(bs, 1, latent, latent, device=device) > 0.5).float()
    masked = torch.randn(bs, 4, latent, latent, device=device)
    text = torch.randn(bs, TEXT_LEN, TEXT_DIM, device=device, dtype=torch.float16)
    return z0, mask, masked, text


def train_step(unet, opt, batch, device):
    z0, mask, masked, text = batch
    bs = z0.shape[0]
    noise = torch.randn_like(z0)
    t = torch.randint(0, 1000, (bs,), device=device)
    # DDPM forward process with the SD scaled-linear schedule: z_t = sqrt(a_bar) z0 + sqrt(1 - a_bar) eps
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, device=device) ** 2
    a_bar = torch.cumprod(1 - betas, 0)[t].view(bs, 1, 1, 1)
    zt = a_bar.sqrt() * z0 + (1 - a_bar).sqrt() * noise
    x = torch.cat([zt, mask, masked], dim=1)  # (B, 9, 64, 64)
    with torch.autocast(device.type, dtype=torch.float16):
        pred = unet(x, t, encoder_hidden_states=text).sample  # (B, 4, 64, 64)
    loss = F.mse_loss(pred.float(), noise)  # ||eps - eps_theta||^2
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss


def count_flops(fn):
    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return counter.get_total_flops()


def bench_train(rank, device, bs, latent, steps, warmup):
    unet = load_unet(device, rank)
    unet.train()
    n_params, n_layers = lora_stats(unet)
    opt = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=1e-4)
    batch = make_batch(bs, latent, device)

    flops = count_flops(lambda: train_step(unet, opt, make_batch(1, latent, device), device))

    for _ in range(warmup):
        train_step(unet, opt, batch, device)
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        loss = train_step(unet, opt, batch, device)
    sync(device)
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
    result = {
        "rank": rank, "lora_params": n_params, "lora_layers": n_layers,
        "flops_per_sample": flops, "batch_size": bs, "timed_steps": steps,
        "samples_per_s": bs * steps / elapsed, "peak_mem_gib": peak,
        "loss_finite": bool(torch.isfinite(loss).item()),
    }
    del unet, opt, batch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return result


@torch.no_grad()
def bench_generate(device, n_images, latent, sampler_steps, timed_images):
    """Cost of one edited image: text-free sampler loop with classifier-free guidance
    (U-Net batch doubled) + one VAE encode of the masked image + one VAE decode."""
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", variant="fp16", torch_dtype=torch.float16).to(device).eval()
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID, subfolder="vae", variant="fp16", torch_dtype=torch.float16).to(device).eval()
    res = latent * 8

    def unet_step(bs):
        x = torch.randn(2 * bs, 9, latent, latent, device=device, dtype=torch.float16)
        t = torch.full((2 * bs,), 500, device=device)
        text = torch.randn(2 * bs, TEXT_LEN, TEXT_DIM, device=device, dtype=torch.float16)
        return unet(x, t, encoder_hidden_states=text).sample

    def vae_encode(bs):
        img = torch.randn(bs, 3, res, res, device=device, dtype=torch.float16)
        return vae.encode(img).latent_dist.mean

    def vae_decode(bs):
        z = torch.randn(bs, 4, latent, latent, device=device, dtype=torch.float16)
        return vae.decode(z).sample

    unet_flops = count_flops(lambda: unet_step(1))
    enc_flops = count_flops(lambda: vae_encode(1))
    dec_flops = count_flops(lambda: vae_decode(1))

    def one_batch(bs):
        vae_encode(bs)
        for _ in range(sampler_steps):
            unet_step(bs)
        vae_decode(bs)

    one_batch(n_images)  # warmup
    sync(device)
    start = time.perf_counter()
    reps = max(1, timed_images // n_images)
    for _ in range(reps):
        one_batch(n_images)
    sync(device)
    elapsed = time.perf_counter() - start

    # VAE encode alone, used for latent precomputation.
    vae_encode(n_images)
    sync(device)
    start = time.perf_counter()
    for _ in range(reps):
        vae_encode(n_images)
    sync(device)
    enc_elapsed = time.perf_counter() - start

    return {
        "images_per_batch": n_images, "sampler_steps": sampler_steps,
        "unet_flops_per_step_per_image_cfg": unet_flops,
        "vae_encode_flops": enc_flops, "vae_decode_flops": dec_flops,
        "flops_per_image": sampler_steps * unet_flops + enc_flops + dec_flops,
        "images_per_s": n_images * reps / elapsed,
        "vae_encodes_per_s": n_images * reps / enc_elapsed,
    }


def plan_hours(plan, train, gen):
    by_rank = {r["rank"]: r for r in train}
    rows, total = [], 0.0
    for run in plan["training_runs"]:
        r = by_rank.get(run["rank"]) or by_rank[plan["lora_rank_default"]]
        samples = run["runs"] * run["steps"] * plan["batch_size"]
        h = samples / r["samples_per_s"] / 3600
        rows.append((run["name"], samples, h))
        total += h
    lp = plan["latent_precompute"]
    encodes = lp["images"] * (1 + lp["regions"]) + lp["train_images"] * lp["regions"] * lp["synthetic_colors_per_region"]
    h = encodes / gen["vae_encodes_per_s"] / 3600
    rows.append(("latent precompute (VAE encodes)", encodes, h))
    total += h
    ev = plan["evaluation"]
    images = ev["models"] * ev["test_images"] * ev["regions"] * ev["target_colors"]
    h = images / gen["images_per_s"] / 3600
    rows.append(("evaluation generation (images)", images, h))
    total += h
    return rows, total


def main():
    device = pick_device(ARGS.device)
    latent = ARGS.resolution // 8
    plan = json.loads(Path(ARGS.plan).read_text())
    print(f"device={device} torch={torch.__version__} resolution={ARGS.resolution} batch={ARGS.batch_size}")
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name())

    train = []
    for rank in ARGS.ranks:
        r = bench_train(rank, device, ARGS.batch_size, latent, ARGS.train_steps, ARGS.warmup)
        train.append(r)
        print(f"rank {rank:>2}: {r['lora_params']/1e6:.2f}M LoRA params in {r['lora_layers']} layers, "
              f"{r['flops_per_sample']/1e12:.3f} TFLOP/sample, {r['samples_per_s']:.2f} samples/s, "
              f"peak {r['peak_mem_gib'] if r['peak_mem_gib'] is None else round(r['peak_mem_gib'], 1)} GiB")

    gen = bench_generate(device, ARGS.gen_batch, latent, ARGS.sampler_steps, ARGS.timed_images)
    print(f"generation: {gen['flops_per_image']/1e12:.2f} TFLOP/image, {gen['images_per_s']:.3f} images/s; "
          f"VAE encode {gen['vae_encode_flops']/1e12:.3f} TFLOP, {gen['vae_encodes_per_s']:.2f}/s")

    rows, total = plan_hours(plan, train, gen)
    print(f"\nPlan on {device} (measured throughput):")
    for name, n, h in rows:
        print(f"  {name:<70} {n:>9,}  {h:7.2f} h")
    print(f"  {'total':<70} {'':>9}  {total:7.2f} h")

    out = {"device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
           "torch": torch.__version__, "args": vars(ARGS), "train": train, "generate": gen,
           "plan_rows": rows, "plan_total_hours": total}
    if ARGS.out:
        Path(ARGS.out).parent.mkdir(parents=True, exist_ok=True)
        Path(ARGS.out).write_text(json.dumps(out, indent=2))
        print("saved", ARGS.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--ranks", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--train-steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--gen-batch", type=int, default=4)
    ap.add_argument("--sampler-steps", type=int, default=30)
    ap.add_argument("--timed-images", type=int, default=8)
    ap.add_argument("--plan", default=str(Path(__file__).resolve().parent.parent / "configs" / "compute_plan.json"))
    ap.add_argument("--out", default="")
    ARGS = ap.parse_args()
    main()
