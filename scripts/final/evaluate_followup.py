"""Evaluate the six explicitly trained follow-up adapters, one saved stage at a time.

Run this yourself on an allocated GPU. It never installs packages or submits jobs.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/final"))
from followup import configure_environment, read_json, run_module

WORK = ROOT / "results/final-followup"
STATE = WORK / "evaluation.json"
REFERENCE = Path("/ocean/projects/cis260330p/shared/runs/wsun8/") / (
    "20261030T181838454135Z_wsun8_evaluate_5297ff9b/cases.json")
GROUPS = {"nored": "NoRed", "p75": "P75"}


def save(state):
    from color_editing.runs import save_json
    save_json(STATE, state)


def adapters(experiment):
    training = read_json(WORK / "manifest.json")["training"]
    result = {}
    for seed in (35, 36, 37):
        item = training.get(f"{experiment}-seed{seed}", {})
        if item.get("status") != "completed":
            raise SystemExit(f"Training is not completed: {experiment} seed {seed}")
        run = Path(item["run"])
        if read_json(run / "run.json")["status"] != "completed":
            raise SystemExit(f"Incomplete training run: {run}")
        weight = run / "adapter/pytorch_lora_weights.safetensors"
        result[f"{GROUPS[experiment]}-r{seed}"] = {
            "adapter": str(weight.parent), "training_run": str(run), "training_seed": seed,
            "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
        }
    return result


def load_state():
    cfg = read_json(ROOT / "configs/main.json")
    if cfg["seed"] != 35 or cfg["split_seed"] != 35:
        raise SystemExit("Evaluation must retain seed 35 and split_seed 35.")
    state = read_json(STATE) if STATE.exists() else {"config": cfg, "groups": {}}
    if state["config"] != cfg:
        raise SystemExit("Evaluation config changed; inspect the saved evaluation first.")
    return state


def verify_images(run, cases, methods):
    from PIL import Image
    from color_editing.evaluate import case_name
    try:
        for method in methods:
            for case in cases:
                with Image.open(run / "images" / method / f"{case_name(case)}.png") as image:
                    if image.size != (512, 512):
                        return False
                    image.verify()
    except (OSError, SyntaxError, ValueError):
        return False
    return True


def metric_stage(state, group, name, module, run):
    item = state["groups"][group]
    previous = item.get(name)
    if previous:
        old = Path(previous)
        if read_json(old / "run.json")["status"] == "completed":
            print(f"Reuse {name}: {old}", flush=True)
            return
        raise SystemExit(f"Recorded {name} is incomplete: {old}")
    pattern = f"*_{module.split('_')[0]}_*"  # identify only runs created by this call
    before = set((ROOT / "results").glob(pattern))
    run_module(module, ["--eval-run", run], f"{group}-{name}")
    found = set((ROOT / "results").glob(pattern)) - before
    # fid_metrics creates kind='fid'; arcface_metrics creates kind='arcface'.
    matches = [p for p in found if read_json(p / "config.json").get("eval_run") == str(run)]
    if len(matches) != 1 or read_json(matches[0] / "run.json")["status"] != "completed":
        raise SystemExit(f"Cannot identify completed {name} run; inspect results/.")
    item[name] = str(matches[0])
    save(state)


def evaluate_group(state, group):
    import torch
    from tqdm import tqdm
    from color_editing import evaluate
    from color_editing.runs import create_run, finish_run, save_json

    chosen = adapters(group)
    cfg = {**state["config"], "adapters": {m: v["adapter"] for m, v in chosen.items()},
           "followup_group": group, "adapter_provenance": chosen}
    print(f"\n=== {GROUPS[group]}: three adapters, fixed evaluation seed 35 ===", flush=True)
    samples, cases = evaluate.test_cases(cfg)
    if cases != read_json(REFERENCE) or len(cases) != 578:
        raise SystemExit("Test cases differ from the published 578-edit protocol; stopped.")
    item = state["groups"].get(group)
    if item is None:
        run = create_run("evaluate", cfg)
        save_json(run / "cases.json", cases)
        item = {"run": str(run), "adapters": chosen, "stages": {}}
        state["groups"][group] = item
        save(state)
    elif item["adapters"] != chosen:
        raise SystemExit("Adapter files or paths changed since evaluation was created.")
    run = Path(item["run"])
    if read_json(run / "config.json") != cfg or read_json(run / "cases.json") != cases:
        raise SystemExit("Saved evaluation config/cases changed; stopped.")
    stages = item["stages"]
    try:
        for method, details in tqdm(chosen.items(), desc=f"{GROUPS[group]} adapters", unit="adapter"):
            if verify_images(run, cases, [method]):
                print(f"Reuse 578 verified images: {method}", flush=True)
                continue
            if stages.get("generation"):
                raise RuntimeError("Previously verified generated images changed; inspect the run "
                                   "before replacing images that already have metrics.")
            # Keep original batch boundaries and seeds; repeat only this method
            # after an interruption, never all successfully generated methods.
            evaluate.generate(cfg, samples, cases, run, [(method, details["adapter"])], resume=True)
            if not verify_images(run, cases, [method]):
                raise RuntimeError(f"Missing/corrupt output images for {method}")
        stages["generation"] = True
        save(state)
        if not stages.get("basic"):
            print("Scoring Lab colors and outside-mask LPIPS; saving B0 outputs.", flush=True)
            (run / "images/B0").mkdir(parents=True, exist_ok=True)
            # score() appends. Preserve an interrupted attempt, then rebuild once.
            metrics = run / "metrics.jsonl"
            if metrics.exists():
                from datetime import datetime, timezone
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                metrics.rename(run / f"metrics-interrupted-{stamp}.jsonl")
            rows = evaluate.score(cfg, samples, cases, run, list(chosen))
            evaluate.summarize(rows, run)
            stages["basic"] = True
            save(state)
        if not verify_images(run, cases, ["B0", *chosen]):
            raise RuntimeError("An output image is missing or invalid after basic scoring.")
        # Keep the existing definition: evaluation completion means generation +
        # Lab/LPIPS; independent CLIP/ArcFace/KID stages have their own records.
        if read_json(run / "run.json")["status"] != "completed":
            finish_run(run, cases=len(cases), methods=["B0", *chosen])
    except BaseException as error:
        finish_run(run, status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   error=f"{type(error).__name__}: {error}")
        raise
    del samples
    torch.cuda.empty_cache()
    if not stages.get("clip"):
        run_module("clip_metrics", ["--eval-run", run], f"{group}-clip")
        stages["clip"] = True
        save(state)
    metric_stage(state, group, "arcface", "arcface_metrics", run)
    metric_stage(state, group, "kid", "fid_metrics", run)
    if not stages.get("figures"):
        arguments = ["--out-dir", run / "figures", "--eval-run", run,
                     "--methods", "B0", *chosen, "--runs",
                     *[f"{method}={details['training_run']}" for method, details in chosen.items()]]
        run_module("plots", arguments, f"{group}-figures")
        stages["figures"] = True
        save(state)
    item["completed"] = True
    save(state)
    print(f"Completed all {GROUPS[group]} evaluation stages: {run}", flush=True)


def run(groups):
    from color_editing.runs import require_gpu
    assets = read_json(WORK / "evaluation-assets.json")
    if assets.get("status") != "completed":
        raise SystemExit("Run prepare_evaluation.py first.")
    require_gpu()
    # Detect missing dependencies before spending time generating images.
    import lpips
    import insightface
    import onnxruntime
    from transformers import CLIPModel, CLIPProcessor
    from torchmetrics.image.fid import NoTrainInceptionV3
    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        raise SystemExit("Use CPU onnxruntime in this cu118 environment, not onnxruntime-gpu.")
    state = load_state()
    for group in groups:
        evaluate_group(state, group)
    print(f"\nSaved evaluation paths and stage status: {STATE}", flush=True)


def status():
    state = load_state()
    for group in GROUPS:
        print(group, json.dumps(state["groups"].get(group, {"status": "not run"}), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status"))
    parser.add_argument("--group", choices=("all", "nored", "p75"), default="all")
    args = parser.parse_args()
    configure_environment()
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    threads = str(max(1, min(4, available)))
    for name in ("ARCFACE_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = threads
    if args.command == "status":
        status()
        return
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / ".evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another follow-up evaluation is already running.")
        run(list(GROUPS) if args.group == "all" else [args.group])


if __name__ == "__main__":
    main()
