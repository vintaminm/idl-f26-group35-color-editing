"""Reuse saved CNN training crops and finish independent validation/test scoring.

Run only after the original full-data workflow has stopped and returned a prompt.
The existing LoRA models, full export recipe and full export journal are preserved.
"""

import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/final"))
from followup import configure_environment
from finish_color import WORK, followup_summary, launch, now, read, save


def main():
    print("Bounded CNN workflow: reuse saved training crops; keep the full held-out splits.", flush=True)
    configure_environment()
    from color_editing.runs import require_gpu
    import torch
    device = require_gpu()
    print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / ".workflow.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("The previous workflow is still running. Wait for it to stop and "
                             "return to the terminal prompt before starting this one.")
        state_path = WORK / "status.json"
        if state_path.exists():
            previous = read(state_path)
            if previous.get("status") == "completed":
                raise SystemExit("The final-color workflow is already verified complete.")
            if previous.get("scope") != "bounded-training-real-crops":
                snapshot = WORK / "status-before-bounded.json"
                if not snapshot.exists():
                    save(snapshot, previous)
        manifest = WORK / "data/dataset-bounded.json"
        training_state = WORK / "cnn/training.json"
        if training_state.exists():
            existing = read(training_state)
            if Path(existing.get("manifest", "")).resolve() != manifest.resolve():
                raise SystemExit("A CNN was already started with another dataset. Preserve it and inspect "
                                 "the saved training metadata before changing dataset scope.")
        state = {"status": "running", "started_utc": now(), "stages": {},
                 "scope": "bounded-training-real-crops", "manifest": str(manifest),
                 "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                 "human_annotation": "pending; return a completed CSV before final analysis",
                 "note": "CNN training uses an explicitly recorded subset. LoRA experiments and "
                         "the fixed validation/test image-ID splits are unchanged."}
        save(state_path, state)
        try:
            followup_summary()
            state["stages"]["followup_tables"] = now()
            save(state_path, state)
            launch("Reuse saved training crops and export held-out validation/test regions",
                   "prepare_color_data_bounded.py")
            dataset = read(manifest)
            state["training_data"] = {"manifest": str(manifest),
                                      "split_counts": dataset["split_region_counts"],
                                      "class_counts": dataset.get("class_counts")}
            state["stages"]["data"] = now()
            save(state_path, state)
            launch("Build the same blinded 184-item annotation package",
                   "human_audit.py", "build-pack")
            state["stages"]["annotation_package"] = now()
            save(state_path, state)
            print("\nANNOTATION PACKAGE READY: results/final-color/human-audit/annotation-pack.zip", flush=True)
            print("Download and label it locally while the CNN continues here.", flush=True)
            launch("Train the CNN on the recorded real-crop training subset",
                   "color_cnn.py", "train", "--manifest", manifest)
            state["stages"]["cnn_training"] = now()
            save(state_path, state)
            launch("Score the held-out real test set and every catalogued edit run",
                   "color_cnn.py", "evaluate", "--manifest", manifest)
            state["stages"]["cnn_evaluation"] = now()
            state.update(status="compute_completed", finished_utc=now())
            save(state_path, state)
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         error=f"{type(exc).__name__}: {exc}", stopped_utc=now())
            save(state_path, state)
            raise
    print(f"\nCOMPUTE STAGES COMPLETED. Status: {state_path}", flush=True)
    print("Return the final annotation CSV to results/final-color/human-audit/responses/.", flush=True)
    print("Then run: python scripts/final/human_audit.py analyze", flush=True)
    print("Then run: python scripts/final/finish_color.py verify", flush=True)
    print("Wait for any concurrent backup step to finish before releasing the allocation.", flush=True)


if __name__ == "__main__":
    main()
