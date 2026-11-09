"""User-run export, annotation-package, and CNN-evaluation workflow.

This script never installs dependencies, submits jobs, or changes shared assets.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "results/final-color"
sys.path.insert(0, str(ROOT / "scripts/final"))
from followup import configure_environment


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    from color_editing.runs import save_json
    save_json(path, value)


def now():
    return datetime.now(timezone.utc).isoformat()


def followup_summary():
    """Aggregate the six completed adapters, retaining seed-level measurements."""
    import csv
    state_path = ROOT / "results/final-followup/evaluation.json"
    if not state_path.exists():
        raise RuntimeError("Finish the six-adapter evaluation before this workflow.")
    state = read(state_path)
    rows = []
    for group in ("nored", "p75"):
        item = state["groups"].get(group, {})
        if not item.get("completed"):
            raise RuntimeError(f"Follow-up evaluation is incomplete: {group}")
        run = Path(item["run"])
        basic = {(r["method"], r["region"], r["target"]): r
                 for r in read(run / "summary.json")}
        clip = {(r["method"], r["region"], r["target"]): r
                for r in read(run / "clip_summary.json")["table"]}
        identity = {(r["method"], r["region"], r["target"]): r
                    for r in read(Path(item["arcface"]) / "arcface_summary.json")}
        kid = {(r["method"], r["region"], r["target"]): r
               for r in read(Path(item["kid"]) / "fid_summary.json")}
        for method, adapter in item["adapters"].items():
            for key, metrics in sorted(basic.items()):
                if key[0] != method:
                    continue
                rows.append({
                    "experiment": group, "seed": adapter["training_seed"],
                    "method": method, "region": key[1], "target": key[2],
                    "n": metrics["n"],
                    "clip_accuracy_percent": 100 * clip[key]["clip_color_accuracy"],
                    "lab_accuracy_percent": 100 * metrics["color_accuracy"],
                    "lpips_outside": metrics["lpips_outside"],
                    "arcface_cosine": identity[key]["identity_cos"],
                    "regional_kid_x1000": 1000 * kid[key]["kid_region"],
                    "evaluation_run": str(run),
                })
    metric_names = ("clip_accuracy_percent", "lab_accuracy_percent", "lpips_outside",
                    "arcface_cosine", "regional_kid_x1000")
    aggregate = []
    groups = {}
    for row in rows:
        groups.setdefault((row["experiment"], row["region"], row["target"]), []).append(row)
    for (experiment, region, target), values in sorted(groups.items()):
        if sorted(v["seed"] for v in values) != [35, 36, 37]:
            raise RuntimeError("Expected exactly seeds 35/36/37 in every summary group.")
        result = {"experiment": experiment, "region": region, "target": target,
                  "training_seeds": 3, "cases_per_seed": values[0]["n"]}
        for name in metric_names:
            numbers = [v[name] for v in values]
            result[f"{name}_mean"] = statistics.mean(numbers)
            result[f"{name}_sample_sd"] = statistics.stdev(numbers)
        aggregate.append(result)
    WORK.mkdir(parents=True, exist_ok=True)
    save(WORK / "followup_summary.json", {
        "created_utc": now(), "source": str(state_path),
        "notes": ["Standard deviations describe three training seeds, not KID subsets.",
                  "These six adapters share the rebuilt follow-up cache.",
                  "CLIP, Lab and the CNN are automatic proxies, not human ground truth."],
        "per_seed": rows, "aggregate": aggregate,
    })
    for name, records in (("followup_per_seed.csv", rows), ("followup_summary.csv", aggregate)):
        destination = WORK / name
        temporary = destination.with_suffix(".csv.partial")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        temporary.replace(destination)
    print(f"Saved three-seed report tables: {WORK / 'followup_summary.csv'}", flush=True)


def launch(label, script, *arguments):
    command = [sys.executable, "-u", str(ROOT / "scripts/final" / script), *map(str, arguments)]
    print(f"\n=== {label} ===\nStarted UTC: {now()}", flush=True)
    print(" ".join(command), flush=True)
    child = subprocess.Popen(command, cwd=ROOT)
    try:
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
    print(f"Completed {label}: {now()}", flush=True)


def run():
    from color_editing.runs import require_gpu
    print("Checking the allocated GPU and current environment...", flush=True)
    device = require_gpu()
    import torch
    print(f"GPU: {torch.cuda.get_device_name(device)} | torch {torch.__version__}", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / ".workflow.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another final-color workflow is already running in this clone.")
        status_path = WORK / "status.json"
        status = {"started_utc": now(), "status": "running", "stages": {},
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "human_annotation": "pending; return a completed CSV before final analysis"}
        save(status_path, status)
        try:
            followup_summary()
            status["stages"]["followup_tables"] = now()
            save(status_path, status)
            launch("Prepare real-region CNN data and blinded annotation images",
                   "prepare_color_data.py")
            status["stages"]["data"] = now()
            save(status_path, status)
            launch("Build the offline annotation website and ZIP", "human_audit.py", "build-pack")
            status["stages"]["annotation_package"] = now()
            save(status_path, status)
            print("\nANNOTATION PACKAGE READY: download the ZIP printed above, unzip it, and open index.html.",
                  flush=True)
            print("You may label images on your computer while the CNN stages continue here.", flush=True)
            launch("Train the small weakly supervised CNN evaluator", "color_cnn.py", "train")
            status["stages"]["cnn_training"] = now()
            save(status_path, status)
            launch("Evaluate real held-out crops and all catalogued edit runs", "color_cnn.py", "evaluate")
            status["stages"]["cnn_evaluation"] = now()
            status.update(status="compute_completed", finished_utc=now())
            save(status_path, status)
        except BaseException as exc:
            status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          error=f"{type(exc).__name__}: {exc}", stopped_utc=now())
            save(status_path, status)
            raise
    print(f"\nCOMPUTE STAGES COMPLETED. Status: {status_path}", flush=True)
    print("Return the annotation CSV to results/final-color/human-audit/responses/.", flush=True)
    print("Then run: python scripts/final/human_audit.py analyze", flush=True)
    print("Final readiness check: python scripts/final/finish_color.py verify", flush=True)
    print("The interactive allocation stays open; use exit after confirming completion.", flush=True)


def verify():
    """Check saved completion records; no GPU, model loading or inference."""
    status_path = WORK / "status.json"
    if not status_path.exists():
        raise SystemExit("Run the final-color compute workflow first.")
    status = read(status_path)
    required = {"followup_tables", "data", "annotation_package", "cnn_training", "cnn_evaluation"}
    if not required.issubset(status.get("stages", {})):
        raise SystemExit("Compute stages are incomplete; inspect results/final-color/status.json.")
    training = read(WORK / "cnn/training.json")
    evaluation = read(WORK / "cnn/evaluation_index.json")
    source = read(WORK / "cnn/source_evaluation.json")
    if any(record.get("status") != "completed" for record in (training, evaluation, source)):
        raise SystemExit("CNN training, real-source scoring or generated-image scoring is incomplete.")
    catalog = read(WORK / "data/evaluation_catalog.json")
    expected = {entry["run_id"] for entry in catalog["runs"]}
    actual = {entry["run_id"] for entry in evaluation["runs"]}
    if actual != expected:
        raise SystemExit("CNN evaluations do not cover the exported run catalog.")
    if evaluation["model_sha256"] != training["best_sha256"]:
        raise SystemExit("CNN evaluation and training refer to different checkpoints.")
    for entry in evaluation["runs"]:
        directory = Path(entry["path"])
        if read(directory / "run.json").get("status") != "completed":
            raise SystemExit(f"Incomplete generated-image CNN scores: {directory}")
    key = read(WORK / "human-audit/key.json")
    reports = sorted((WORK / "human-audit/analysis").glob("*/summary.json"))
    ready = None
    for path in reversed(reports):
        report = read(path)
        if report.get("pack_id") != key["pack_id"]:
            continue
        if (report.get("annotators") and not report.get("missing_cnn_item_ids")
                and all(person.get("complete") for person in report["annotators"].values())):
            ready = path
            break
    if ready is None:
        raise SystemExit("Human audit is pending. Finish all annotation items, return the CSV, "
                         "and run python scripts/final/human_audit.py analyze.")
    status.update(status="completed", human_annotation="completed", audit_summary=str(ready),
                  verified_utc=now())
    save(status_path, status)
    print("FINAL_COLOR_WORK_COMPLETE: six-adapter tables, CNN train/test/edit metrics, "
          "and the completed human audit are saved.")
    print(f"Human analysis: {ready.parent / 'report.md'}")
    print(f"Readiness record: {status_path}")
    print("Back up the complete results and code before PSC becomes unavailable.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "summarize", "status", "verify"))
    args = parser.parse_args()
    print(f"Final color-evaluation workflow: {args.command}", flush=True)
    configure_environment()
    threads = str(max(1, min(4, len(os.sched_getaffinity(0)))))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = threads
    if args.command == "run":
        run()
    elif args.command == "summarize":
        followup_summary()
    elif args.command == "verify":
        verify()
    else:
        path = WORK / "status.json"
        print(path.read_text() if path.exists() else "No final-color workflow has been run.")


if __name__ == "__main__":
    main()
