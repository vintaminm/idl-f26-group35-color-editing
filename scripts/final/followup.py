"""Prepare one fixed cache, then explicitly run one final-report replicate.

Nothing is submitted to Slurm. Run this from an allocated GPU terminal.
"""

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "results/final-followup"
CACHE = WORK / "cache"
MANIFEST = WORK / "manifest.json"
CONFIGS = {"nored": "final-nored.json", "p75": "final-p75.json"}
CACHE_PARTS = (
    ("train-0", "train", ["--shard", "0", "--num-shards", "2"]),
    ("train-1", "train", ["--shard", "1", "--num-shards", "2"]),
    ("val", "val", ["--limit", "500"]),
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_manifest(state):
    from color_editing.runs import save_json
    save_json(MANIFEST, state)


def recipe():
    # Keep posterior sampling, shard boundaries and recoloring fixed across seeds.
    files = ["configs/main.json", "configs/final-nored.json", "configs/final-p75.json",
             "src/color_editing/cache.py", "src/color_editing/train.py", "src/color_editing/data.py",
             "src/color_editing/pipeline.py"]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}


def load_state(create=False):
    if MANIFEST.exists():
        state = read_json(MANIFEST)
        if state["recipe"] != recipe():
            raise SystemExit("Cache/config recipe changed. Keep the existing campaign intact; "
                             "inspect it before creating another cache.")
        return state
    if not create:
        raise SystemExit("Run the prepare command first.")
    state = {"created_utc": datetime.now(timezone.utc).isoformat(),
             "recipe": recipe(), "base_config": read_json(ROOT / "configs/main.json"),
             "training_configs": {k: read_json(ROOT / "configs" / v)
                                  for k, v in CONFIGS.items()},
             "cache": {}, "training": {}}
    write_manifest(state)
    return state


def configure_environment():
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise SystemExit("Use this repository's .venv/bin/python.")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    os.environ.update(PYTHONPATH=str(ROOT / "src"), PYTHONNOUSERSITE="1",
                      PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", HF_HOME=str(ROOT / "results/_hf_cache"),
                      TORCH_HOME=str(ROOT / "results/_torch_cache"))


def gpu_preflight():
    from color_editing.runs import require_gpu
    import torch
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from peft import LoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer
    from tqdm.auto import tqdm

    require_gpu()
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}", flush=True)
    shared = Path(read_json(ROOT / "configs/main.json")["shared_root"])
    for path in (shared / "models/sd15-inpainting", shared / "data/CelebAMask-HQ"):
        if not path.is_dir():
            raise SystemExit(f"Missing shared asset: {path}")


def run_module(module, args, label):
    """Save readable stdout; stderr stays on the terminal for dynamic tqdm bars."""
    logs = WORK / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "-m", f"color_editing.{module}", *map(str, args)]
    print(f"\n=== {label} ===\nLog: {logs / (label + '.log')}", flush=True)
    with (logs / f"{label}.log").open("a") as log:
        log.write("\n" + json.dumps(command) + "\n")
        log.flush()
        with subprocess.Popen(command, stdout=subprocess.PIPE, text=True, bufsize=1) as child:
            try:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = child.wait()
            except BaseException:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def prepare():
    gpu_preflight()
    state = load_state(create=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    for index, (name, split, options) in enumerate(CACHE_PARTS, 1):
        target = CACHE / f"{name}.pt"
        metadata = target.with_suffix(".json")
        if name in state["cache"]:
            if not target.is_file() or not metadata.is_file():
                raise SystemExit(f"Recorded cache is missing: {target}")
            if target.stat().st_size != state["cache"][name]["bytes"]:
                raise SystemExit(f"Recorded cache size changed: {target}")
            print(f"[{index}/3] Reuse {target}", flush=True)
            continue
        if target.exists() or metadata.exists():
            raise SystemExit(f"Unrecorded cache exists; inspect before reuse: {target}")
        # Interrupted writes cannot look like a completed final cache.
        temporary = CACHE / f"{name}.partial.pt"
        run_module("cache", ["--config", ROOT / "configs/main.json", "--split", split,
                             *options, "--out", temporary], f"cache-{name}")
        info = read_json(temporary.with_suffix(".json"))
        if info["config"] != state["base_config"] or info["region_samples"] <= 0:
            raise SystemExit(f"Unexpected cache metadata: {temporary}")
        temporary.replace(target)
        temporary.with_suffix(".json").replace(metadata)
        state["cache"][name] = {"path": str(target), "bytes": target.stat().st_size,
                                  "images": info["images"], "regions": info["region_samples"]}
        write_manifest(state)
    print(f"\nCache ready. Manifest: {MANIFEST}", flush=True)
    print("Next: python scripts/final/followup.py train --experiment nored --seed 35", flush=True)


def check_time():
    # V100 measurements were about 101--103 min; allow startup and a margin.
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return
    try:
        remaining = subprocess.check_output(
            ["squeue", "-h", "-j", job, "-o", "%L"], text=True, timeout=10).strip()
        days, clock = remaining.split("-", 1) if "-" in remaining else ("0", remaining)
        fields = list(map(int, clock.split(":")))
        seconds = int(days) * 86400 + sum(n * 60 ** i for i, n in enumerate(reversed(fields)))
    except (OSError, ValueError, subprocess.SubprocessError):
        raise SystemExit("Cannot read allocation time remaining. Check squeue before starting training.")
    print(f"Allocation time remaining: {remaining}", flush=True)
    if seconds < 125 * 60:
        raise SystemExit("Less than 125 minutes remain. Keep the completed cache/results and "
                         "start this replicate in the next allocation.")


def train(experiment, seed):
    state = load_state()
    for name, _, _ in CACHE_PARTS:
        saved = state["cache"].get(name)
        path = CACHE / f"{name}.pt"
        if not saved or not path.is_file() or path.stat().st_size != saved["bytes"]:
            raise SystemExit("Cache incomplete or changed. Run prepare first.")
    key = f"{experiment}-seed{seed}"
    previous = state["training"].get(key)
    if previous and previous["status"] == "completed":
        path = Path(previous["run"])
        if read_json(path / "run.json")["status"] != "completed" or not (
                path / "adapter/pytorch_lora_weights.safetensors").is_file():
            raise SystemExit(f"Recorded adapter is missing or incomplete: {path}")
        print(f"Already completed: {path}", flush=True)
        return
    gpu_preflight()
    check_time()
    pattern = f"*_{getpass.getuser()}_train-{experiment}-seed{seed}_*"
    before = set((ROOT / "results").glob(pattern))
    args = ["--config", ROOT / "configs" / CONFIGS[experiment], "--seed", seed,
            "--train-cache", CACHE / "train-0.pt", CACHE / "train-1.pt",
            "--val-cache", CACHE / "val.pt"]
    state["training"][key] = {"status": "running", "cache_manifest": str(MANIFEST)}
    write_manifest(state)
    try:
        run_module("train", args, key)
    except BaseException as error:
        state["training"][key].update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}")
        raise
    finally:
        created = set((ROOT / "results").glob(pattern)) - before
        if len(created) == 1:
            run = created.pop()
            record = read_json(run / "run.json")
            state["training"][key] = {"run": str(run), "status": record["status"],
                                       "cache_manifest": str(MANIFEST)}
        write_manifest(state)
    completed = state["training"].get(key, {})
    if completed.get("status") != "completed":
        raise SystemExit("Training did not record completion. Inspect the terminal and run.json.")
    print(f"\nCompleted {key}: {completed['run']}", flush=True)
    print("Adapter is saved separately; shared base weights were only read.", flush=True)


def status():
    state = load_state()
    print(f"Campaign: {MANIFEST}")
    for name, _, _ in CACHE_PARTS:
        item = state["cache"].get(name)
        print(f"cache/{name}: {str(item['regions']) + ' regions' if item else 'not prepared'}")
    for experiment in CONFIGS:
        for seed in (35, 36, 37):
            key = f"{experiment}-seed{seed}"
            item = state["training"].get(key, {})
            print(f"{key}: {item.get('status', 'not run')} {item.get('run', '')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Build/reuse the fixed seed-35 B0 latent cache")
    one = commands.add_parser("train", help="Run one explicit 10,000-step replicate")
    one.add_argument("--experiment", choices=CONFIGS, required=True)
    one.add_argument("--seed", type=int, choices=(35, 36, 37), required=True)
    commands.add_parser("status", help="Print saved cache and training paths; no GPU work")
    args = parser.parse_args()
    configure_environment()
    if args.command == "status":
        status()
        return
    # One writer per campaign, including while its child is running.
    import fcntl
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another followup command is still running in this clone.")
        if args.command == "prepare":
            prepare()
        elif args.command == "train":
            train(args.experiment, args.seed)


if __name__ == "__main__":
    main()
