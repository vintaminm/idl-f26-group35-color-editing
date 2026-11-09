"""Reuse completed training crops, then finish full validation/test crop exports.

Run after stopping the full exporter and waiting for its prompt to return. This
separate scope preserves that export's code, recipe, journal and existing crops.
There is no model inference, package installation, job submission or download.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import stat
import sys


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "results/final-color"
DATA = WORK / "data"
TRAIN_CAP = 20000
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_color_data as full


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def journal_records(path):
    """Read committed rows only, preserving the full export's journal byte-for-byte."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"No completed training-crop journal exists: {path}")
    lines = path.read_bytes().splitlines(keepends=True)
    records, ignored_last = {}, False
    for index, raw in enumerate(lines):
        if index == len(lines) - 1 and not raw.endswith(b"\n"):
            print("Ignoring the uncommitted final journal fragment; the source journal is unchanged.", flush=True)
            ignored_last = True
            break
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Malformed committed journal row {index + 1}; inspect {path}.") from exc
        identity = row["sample_id"]
        if identity in records and records[identity] != row:
            raise RuntimeError(f"Conflicting committed crop metadata: {identity}")
        records[identity] = row
    return list(records.values()), ignored_last


def freeze_selection():
    from color_editing.data import split_ids
    from color_editing.runs import save_json

    selection_path = DATA / "bounded-selection.json"
    if selection_path.exists():
        selection = full.read_json(selection_path)
        if selection.get("selection_version") != 1 or selection.get("training_cap") != TRAIN_CAP:
            raise SystemExit("The existing bounded selection uses another recipe; preserve it and inspect first.")
        if fingerprint(selection["train_records"]) != selection["train_records_sha256"]:
            raise SystemExit("Frozen bounded training metadata changed; inspect before continuing.")
        return selection
    journal = DATA / "records.partial.jsonl"
    records, ignored = journal_records(journal)
    partitions = {name: set(ids) for name, ids in split_ids(35).items()}
    for row in records:
        if row["split"] not in partitions or row["image_id"] not in partitions[row["split"]]:
            raise RuntimeError(f"Unexpected original-photo split in {row['sample_id']}")
        if row["region"] not in {"hair", "cloth"} or row["label"] not in full.CLASSES:
            raise RuntimeError(f"Invalid region or label in {row['sample_id']}")
        expected_image = f"crops/{row['split']}/{row['sample_id']}.jpg"
        if row["image"] != expected_image:
            raise RuntimeError(f"Unsafe crop path: {row['image']}")
    completed_train = [row for row in records if row["split"] == "train"]
    picked = completed_train[:TRAIN_CAP]
    if not picked:
        raise SystemExit("The full export has no committed training crops to reuse.")
    counts = dict(Counter(row["label"] for row in picked))
    missing = sorted(set(full.CLASSES) - set(counts))
    if missing:
        raise SystemExit(f"The bounded training selection does not cover all ten colors: missing {missing}.")
    selection = {
        "selection_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_seed": 35, "training_cap": TRAIN_CAP,
        "training_selection": "All committed training regions in fixed original-image order, up to the cap; no class balancing.",
        "observed_completed_training_regions": len(completed_train),
        "source_journal": str(journal), "source_journal_sha256": full.digest(journal),
        "ignored_uncommitted_final_fragment": ignored,
        "train_records_sha256": fingerprint(picked), "train_class_counts": counts,
        "train_records": picked,
        "available_validation_test_records": [row for row in records if row["split"] in {"val", "test"}],
    }
    save_json(selection_path, selection)
    print(f"Frozen bounded training selection: {len(picked):,} completed real regions; class counts {counts}", flush=True)
    return selection


def bounded_recipe(selection):
    return {
        "schema_version": 1, "scope": "completed-training-crops-full-validation-test-v1",
        "split_seed": 35, "crop_size": full.SIZE, "classes": full.CLASSES,
        "jpeg_quality": 95, "min_mask_fraction": 0.01, "label_source": "Lab weak labels",
        "training_cap": TRAIN_CAP, "selected_training_regions": len(selection["train_records"]),
        "train_records_sha256": selection["train_records_sha256"],
        "code_sha256": {name: full.digest(ROOT / name) for name in (
            "scripts/final/prepare_color_data_bounded.py", "scripts/final/prepare_color_data.py",
            "src/color_editing/data.py", "src/color_editing/clip_metrics.py")},
    }


def validate_training_files(records):
    from tqdm import tqdm
    for row in tqdm(records, desc="Check committed training files", unit="crop", dynamic_ncols=True):
        path = DATA / row["image"]
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            raise RuntimeError(f"A committed training crop is missing or empty: {path}")
    selected = random.Random(35).sample(records, min(16, len(records)))
    for row in tqdm(selected, desc="Sample-check crop SHA256", unit="crop", dynamic_ncols=True):
        if not full.valid_crop(DATA / row["image"], row["sha256"]):
            raise RuntimeError(f"Sampled committed crop failed its hash or image check: {row['sample_id']}")
    print("Training reuse: all selected files exist and are nonempty; 16 sampled crops have full SHA/image checks.", flush=True)


def export_bounded_dataset(data_root):
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    from color_editing.clip_metrics import region_crop
    from color_editing.data import _load_mask, _mask_path, describe_sample, split_ids
    from color_editing.runs import save_json

    DATA.mkdir(parents=True, exist_ok=True)
    selection = freeze_selection()
    recipe = bounded_recipe(selection)
    recipe_path = DATA / "bounded-recipe.json"
    if recipe_path.exists() and full.read_json(recipe_path) != recipe:
        raise SystemExit("Bounded export code or frozen selection changed; keep the current export intact.")
    save_json(recipe_path, recipe)
    destination = DATA / "dataset-bounded.json"
    if destination.exists():
        previous = full.read_json(destination)
        if previous.get("status") == "completed" and previous.get("recipe") == recipe:
            print(f"Bounded dataset already completed: {len(previous['records']):,} records.", flush=True)
            return previous
        raise SystemExit("An existing bounded dataset uses another recipe; inspect before replacing it.")

    train = selection["train_records"]
    validate_training_files(train)
    journal = DATA / "bounded-records.partial.jsonl"
    resumed = full.recover_records(journal, DATA)
    originals = {row["sample_id"]: row for row in selection["available_validation_test_records"]}
    records = {**originals, **resumed}
    partitions = split_ids(35)
    workers = max(1, min(4, len(os.sched_getaffinity(0)), int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    ordered = list(train)
    print(f"Export only the full validation/test splits: 4,000 photographs, {workers} workers.", flush=True)
    with journal.open("a") as log, ThreadPoolExecutor(max_workers=workers) as pool:
        for split in ("val", "test"):
            ids = partitions[split]

            def prepare_image(image_id):
                image, rows = None, []
                for region in ("hair", "cloth"):
                    identity = f"{image_id:05d}-{region}"
                    if identity in records:
                        row = records[identity]
                        if row["split"] != split:
                            raise RuntimeError(f"Unexpected resumed split: {identity}")
                        path = DATA / row["image"]
                        info = path.stat()
                        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
                            raise RuntimeError(f"Resumed validation/test crop is missing: {path}")
                    else:
                        mask_path = _mask_path(data_root, image_id, region)
                        if not mask_path.is_file():
                            continue
                        mask = _load_mask(mask_path)
                        fraction = float(np.mean(np.asarray(mask) > 127))
                        if fraction < 0.01:
                            continue
                        if image is None:
                            with Image.open(data_root / "CelebA-HQ-img" / f"{image_id}.jpg") as original:
                                image = original.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
                        stats = describe_sample({"image": image, "mask": mask,
                                                 "image_id": image_id, "region": region})
                        relative = f"bounded-crops/{split}/{identity}.jpg"
                        checksum = full.save_image(region_crop(image, mask, size=full.SIZE), DATA / relative)
                        row = {"sample_id": identity, "image_id": image_id, "region": region,
                               "split": split, "label": full.canonical(stats["source_color"]),
                               "weak_label": stats["source_color"], "image": relative,
                               "mask_fraction": fraction, "sha256": checksum}
                    rows.append(row)
                return rows

            count = 0
            progress = tqdm(total=len(ids), desc=f"Bounded export {split}", unit="image",
                            dynamic_ncols=True, mininterval=1.0)
            for start in range(0, len(ids), 64):
                for image_records in pool.map(prepare_image, ids[start:start + 64]):
                    for row in image_records:
                        if row["sample_id"] not in resumed:
                            resumed[row["sample_id"]] = row
                            records[row["sample_id"]] = row
                            log.write(json.dumps(row, allow_nan=False) + "\n")
                            log.flush()
                        ordered.append(row)
                        count += 1
                    progress.update(1)
                    progress.set_postfix(regions=count, refresh=False)
            progress.close()

    seen = set()
    split_sets = {name: set(ids) for name, ids in partitions.items()}
    for row in ordered:
        if row["sample_id"] in seen or row["image_id"] not in split_sets[row["split"]]:
            raise RuntimeError(f"Duplicate or cross-split bounded crop: {row['sample_id']}")
        seen.add(row["sample_id"])
    counts = {split: dict(Counter(row["label"] for row in ordered if row["split"] == split))
              for split in partitions}
    manifest = {
        **recipe, "status": "completed", "recipe": recipe,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "CelebAMask-HQ", "data_root": str(data_root),
        "selection_file": "bounded-selection.json",
        "split_image_counts": {split: len({row['image_id'] for row in ordered if row['split'] == split})
                               for split in partitions},
        "split_region_counts": dict(Counter(row["split"] for row in ordered)),
        "class_counts": counts, "records": ordered,
        "limitations": [
            "Training uses the frozen prefix of completed real-region crops, not all 37,971 training regions.",
            "All original validation and test image IDs were scanned with the same 1% minimum mask rule.",
            "Colors are automatic Lab weak labels; held-out agreement is not human-ground-truth accuracy.",
            "All ten training classes are present, but class imbalance and very rare colors limit the evaluator.",
            "blond and yellow share a prototype and are merged to yellow internally.",
            "Original-photo splits are disjoint; identities are not guaranteed disjoint.",
        ],
    }
    save_json(destination, manifest)
    (DATA / "dataset-bounded.sha256").write_text(full.digest(destination) + "  dataset-bounded.json\n")
    print(f"Bounded dataset complete: {len(train):,} train regions; all qualifying val/test regions retained.", flush=True)
    print("Actual split region counts:", manifest["split_region_counts"], flush=True)
    print("Class counts:", json.dumps(counts, sort_keys=True), flush=True)
    return manifest


def main():
    print("Starting bounded color export: reuse committed training crops; no full-training re-export.", flush=True)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    argparse.ArgumentParser(description=__doc__).parse_args()
    if Path("/ocean/projects/cis260330p").exists():
        if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_ACCOUNT") != "cis260330p":
            raise SystemExit("Use the current allocated compute-node terminal under -A cis260330p.")
    sys.path.insert(0, str(ROOT / "src"))
    from color_editing.runs import save_json
    shared = Path(full.read_json(ROOT / "configs/main.json")["shared_root"])
    with full.export_lock():
        export_bounded_dataset(shared / "data/CelebAMask-HQ")
        catalog = full.export_catalog(shared)
        full.export_audit(shared / "data/CelebAMask-HQ", catalog)
        save_json(WORK / "bounded-export-status.json", {
            "status": "completed", "scope": "bounded training; full validation/test; full edit catalog",
            "dataset_manifest": str(DATA / "dataset-bounded.json"),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
    print("Bounded color-data preparation complete.", flush=True)


if __name__ == "__main__":
    main()
