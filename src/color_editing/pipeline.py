"""Small real-data B1/LoRA flows. GPU work only starts on explicit calls."""

import gc
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

from .data import COLOR_RGB, contact_sheet, describe_sample, recolor_lab
from .runs import append_jsonl, finish_run, require_gpu, save_json

LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


def prompt_for(region, color):
    if region not in ("hair", "cloth") or color not in COLOR_RGB:
        raise ValueError("Use a hair/cloth region and a color from COLOR_RGB.")
    return (f"a photo of a person with {color} hair" if region == "hair"
            else f"a photo of a person wearing {color} clothes")


def target_color_for(sample, requested):
    if requested not in COLOR_RGB:
        raise ValueError(f"Unknown target color: {requested}")
    source = describe_sample(sample)["source_color"]
    return requested if requested != source else ("blue" if source == "red" else "red")


def load_pipeline(model_dir, adapter_dir=None):
    """Read the local fp16 inpainting model; never fetch weights from the Hub."""
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    device = require_gpu()
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "unet/config.json").read_text())
    if config["in_channels"] != 9:
        raise ValueError("Expected the proposal's 9-channel inpainting U-Net.")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        str(model_dir), variant="fp16", dtype=torch.float16,
        use_safetensors=True, local_files_only=True,
    ).to(device)
    if adapter_dir is not None:
        pipe.load_lora_weights(str(adapter_dir), local_files_only=True)
    return pipe


def _outside_mae(original, edited, mask):
    outside = np.asarray(mask) == 0
    difference = np.abs(np.asarray(original, dtype=np.float32)
                        - np.asarray(edited, dtype=np.float32)) / 255.0
    return float(difference[outside].mean()) if outside.any() else None


def compare_edits(samples, model_dir, run_dir, config, adapter_dir=None):
    """Save B0/B1, plus a freshly loaded LoRA when supplied, on identical cases.

    Raw diffusion outputs and composites use separate files/records. Composite
    background preservation is postprocessing, not evidence of learned behavior.
    """
    import torch

    run_dir = Path(run_dir)
    pipe = None
    try:
        require_gpu()
        seed = int(config["seed"])
        inference = config["inference"]
        cases = []
        if not samples:
            raise ValueError("Select at least one sample.")
        for sample in samples:
            meta = describe_sample(sample)
            color = target_color_for(sample, config["target_color"])
            prompt = prompt_for(sample["region"], color)
            case = run_dir / "images" / f"{sample['image_id']:05d}-{sample['region']}"
            case.mkdir(parents=True, exist_ok=False)
            for name in ("image", "mask", "edit_mask"):
                sample[name].save(case / f"{name}.png")
            b0 = recolor_lab(sample["image"], sample["mask"], color)
            b0.save(case / "B0.png")
            record = {**meta, "target_color": color, "prompt": prompt,
                      "seed": seed + sample["image_id"], "split": "val",
                      "mask_dilation": config["mask_dilation"],
                      "case_path": str(case.relative_to(run_dir))}
            append_jsonl(run_dir / "images.jsonl", {
                **record, "method": "B0", "output": str((case / "B0.png").relative_to(run_dir)),
                "outside_original_mask_mae_01": _outside_mae(sample["image"], b0, sample["mask"]),
                "postprocessed": False,
            })
            cases.append((sample, case, record))
        save_json(run_dir / "samples.json", [record for _, _, record in cases])
        pipe = load_pipeline(model_dir)
        methods = ["B1"] + (["LoRA"] if adapter_dir is not None else [])
        for method in methods:
            if method == "LoRA":
                pipe.load_lora_weights(str(adapter_dir), local_files_only=True)
            for sample, case, record in cases:
                generator = torch.Generator(device="cuda").manual_seed(record["seed"])
                print(f"{method}: image {sample['image_id']} -> {record['target_color']}", flush=True)
                with torch.inference_mode():
                    output = pipe(
                        prompt=record["prompt"], image=sample["image"],
                        mask_image=sample["edit_mask"], height=512, width=512,
                        num_inference_steps=int(inference["steps"]),
                        guidance_scale=float(inference["guidance_scale"]),
                        strength=1.0, generator=generator,
                    )
                raw = output.images[0].convert("RGB")
                composite = Image.composite(raw, sample["image"], sample["mask"])
                flagged = (bool(output.nsfw_content_detected[0])
                           if output.nsfw_content_detected is not None else None)
                if flagged:
                    print("Safety checker flagged this output; the saved result may be black.", flush=True)
                for variant, image in (("raw", raw), ("composite", composite)):
                    filename = f"{method}-{variant}.png"
                    image.save(case / filename)
                    append_jsonl(run_dir / "images.jsonl", {
                        **record, "method": method, "variant": variant,
                        "output": str((case / filename).relative_to(run_dir)),
                        "outside_original_mask_mae_01": _outside_mae(sample["image"], image, sample["mask"]),
                        "postprocessed": variant == "composite", "safety_flagged": flagged,
                        "inference": inference,
                    })
        boards = []
        for sample, case, _ in cases:
            names = ["image", "mask", "B0", "B1-raw", "B1-composite"]
            if adapter_dir is not None:
                names += ["LoRA-raw", "LoRA-composite"]
            pictures = []
            for name in names:
                with Image.open(case / f"{name}.png") as image:
                    pictures.append(image.convert("RGB"))
            board = contact_sheet(pictures, names)
            path = case / "comparison.png"
            board.save(path)
            boards.append(path)
        finish_run(run_dir, cases=len(cases), methods=methods,
                   adapter=str(adapter_dir) if adapter_dir else None,
                   gpu=torch.cuda.get_device_name(0), cuda_version=torch.version.cuda)
        return boards
    except BaseException as error:
        finish_run(run_dir, status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   error=f"{type(error).__name__}: {error}")
        raise
    finally:
        pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _save_adapter(unet, destination, step, config):
    from diffusers import StableDiffusionInpaintPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import get_peft_model_state_dict

    state = get_peft_model_state_dict(unet)
    state = {key: value.detach().float().cpu() for key, value in state.items()}
    StableDiffusionInpaintPipeline.save_lora_weights(
        str(destination), unet_lora_layers=convert_state_dict_to_diffusers(state),
        safe_serialization=True,
    )
    save_json(Path(destination) / "training.json", {
        "step": step, "training": config["training"], "seed": config["seed"],
        "lora_targets": LORA_TARGETS, "base_model": "sd15-inpainting",
        "checkpoint_kind": "adapter_only_no_optimizer_resume",
    })


def train_lora(samples, model_dir, run_dir, config):
    """Reconstruction-only, batch-one pilot; only attention LoRA is optimized.

    The small selected set is encoded once in RAM. No full-data embedding cache,
    augmentation, synthetic supervision, or optimizer-resume is introduced here.
    """
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler
    from peft import LoraConfig

    run_dir = Path(run_dir)
    if (run_dir / "loss.jsonl").exists():
        raise ValueError("Create a fresh run before starting training again.")
    pipe = optimizer = None
    cache = []
    try:
        device = require_gpu()
        training = config["training"]
        if training["supervision"] != "reconstruction":
            raise ValueError("This pilot supports reconstruction supervision only.")
        steps, rank = int(training["steps"]), int(training["rank"])
        if not samples or steps < 1 or rank < 1 or int(training["save_every"]) < 1:
            raise ValueError("Select samples and positive training parameters.")
        torch.manual_seed(int(config["seed"]))
        rng = random.Random(int(config["seed"]))
        generator = torch.Generator(device=device).manual_seed(int(config["seed"]))
        save_json(run_dir / "samples.json", [
            {**describe_sample(s), "split": "train", "prompt": prompt_for(s["region"], describe_sample(s)["source_color"])}
            for s in samples
        ])
        pipe = load_pipeline(model_dir)
        pipe.vae.requires_grad_(False).eval().to(dtype=torch.float32)
        pipe.text_encoder.requires_grad_(False).eval()
        if pipe.safety_checker is not None:
            pipe.safety_checker.to("cpu")
        unet = pipe.unet
        unet.requires_grad_(False)
        scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        if scheduler.config.prediction_type != "epsilon":
            raise ValueError("Expected epsilon noise prediction for this SD 1.5 pilot.")
        print(f"Encoding {len(samples)} training samples and prompts in RAM...", flush=True)
        with torch.no_grad():
            for index, sample in enumerate(samples):
                pixels = torch.from_numpy(np.asarray(sample["image"], dtype=np.float32).copy())
                pixels = pixels.permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0
                mask = torch.from_numpy((np.asarray(sample["edit_mask"]) > 0).astype(np.float32))
                mask = mask[None, None].to(device)
                scale = pipe.vae.config.scaling_factor
                latent = pipe.vae.encode(pixels).latent_dist.sample(generator=generator) * scale
                masked = pipe.vae.encode(pixels * (1 - mask)).latent_dist.sample(generator=generator) * scale
                ids = pipe.tokenizer(
                    prompt_for(sample["region"], describe_sample(sample)["source_color"]),
                    padding="max_length", max_length=pipe.tokenizer.model_max_length,
                    truncation=True, return_tensors="pt",
                ).input_ids.to(device)
                text = pipe.text_encoder(ids)[0]
                small_mask = F.interpolate(mask, size=latent.shape[-2:], mode="nearest")
                cache.append(tuple(x.detach().to(device="cpu", dtype=torch.float16)
                                   for x in (latent, small_mask, masked, text)))
                print(f"  encoded {index + 1}/{len(samples)}", flush=True)
        # VAE/text conditioning is fixed for this short pilot; free their GPU memory.
        pipe.vae.to("cpu")
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        unet.add_adapter(LoraConfig(
            r=rank, lora_alpha=rank, init_lora_weights="gaussian", target_modules=LORA_TARGETS,
        ))
        trainable = []
        for name, parameter in unet.named_parameters():
            if parameter.requires_grad:
                if "lora_" not in name:
                    raise RuntimeError(f"Unexpected trainable base parameter: {name}")
                parameter.data = parameter.data.float()
                trainable.append(parameter)
        if not trainable:
            raise RuntimeError("No LoRA parameters were added.")
        unet.enable_gradient_checkpointing()
        unet.train()
        optimizer = torch.optim.AdamW(trainable, lr=float(training["learning_rate"]))
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
        torch.cuda.reset_peak_memory_stats()
        parameter_count = sum(p.numel() for p in trainable)
        print(f"Training {parameter_count:,} LoRA parameters; batch size 1, {steps} steps.", flush=True)
        started = time.perf_counter()
        for step in range(1, steps + 1):
            latent, mask, masked, text = [x.to(device) for x in cache[rng.randrange(len(cache))]]
            noise = torch.randn(latent.shape, generator=generator, device=device, dtype=latent.dtype)
            timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), generator=generator, device=device)
            noisy = scheduler.add_noise(latent, noise, timestep)
            model_input = torch.cat([noisy, mask, masked], dim=1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = unet(model_input, timestep, encoder_hidden_states=text).sample
                loss = F.mse_loss(prediction.float(), noise.float())
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss; stopping training.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(grad_norm).item():
                raise RuntimeError("Non-finite gradients; stopping training.")
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            record = {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad_norm),
                      "elapsed_seconds": time.perf_counter() - started}
            append_jsonl(run_dir / "loss.jsonl", record)
            print(f"step {step:03d}/{steps}  loss={record['loss']:.5f}  grad={record['grad_norm']:.4f}", flush=True)
            if step % int(training["save_every"]) == 0 and step < steps:
                _save_adapter(unet, run_dir / "checkpoints" / f"step-{step:06d}", step, config)
        adapter = run_dir / "adapter"
        _save_adapter(unet, adapter, steps, config)
        finish_run(run_dir, steps=steps, trainable_parameters=parameter_count,
                   peak_cuda_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                   adapter="adapter", supervision="reconstruction", batch_size=1,
                   gpu=torch.cuda.get_device_name(0), cuda_version=torch.version.cuda)
        return adapter
    except BaseException as error:
        finish_run(run_dir, status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   error=f"{type(error).__name__}: {error}")
        raise
    finally:
        pipe = optimizer = None
        cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
