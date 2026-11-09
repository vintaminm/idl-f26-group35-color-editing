"""Export real-region crops and a blinded human audit; no model runs or downloads.

Run explicitly from the project environment on an allocated compute node. Originals
and shared results are read only. Every output stays below results/final-color/.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "results/final-color"
CLASSES = ["black", "brown", "yellow", "red", "blue", "green", "purple", "pink", "white", "gray"]
SIZE = 128
SCHEMA = 1


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def canonical(name):
    return "yellow" if name == "blond" else name


def recipe():
    return {
        "schema_version": SCHEMA, "split_seed": 35, "crop_size": SIZE,
        "jpeg_quality": 95, "min_mask_fraction": 0.01,
        "classes": CLASSES, "label_source": "Lab weak labels",
        "code_sha256": {name: digest(ROOT / name) for name in (
            "src/color_editing/data.py", "src/color_editing/clip_metrics.py",
            "scripts/final/prepare_color_data.py")},
    }


@contextmanager
def export_lock():
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / ".prepare.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("Another color-data export is running; let it finish.") from exc
        yield


def save_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.jpg")
    image.convert("RGB").save(temporary, format="JPEG", quality=95, subsampling=0)
    temporary.replace(path)
    return digest(path)


def valid_crop(path, expected_hash, size=SIZE):
    from PIL import Image
    try:
        if digest(path) != expected_hash:
            return False
        with Image.open(path) as image:
            if image.size != (size, size) or image.mode != "RGB":
                return False
            image.verify()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def recover_records(journal, target):
    """Only a truncated last journal line is recoverable; earlier corruption stops."""
    from tqdm import tqdm
    if not journal.exists():
        return {}
    lines = journal.read_text().splitlines()
    good = {}
    for index, line in enumerate(tqdm(lines, desc="Check exported crops", unit="crop", dynamic_ncols=True)):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise RuntimeError(f"Malformed journal line {index + 1}: {journal}")
            print("Recovering an incomplete final journal entry.", flush=True)
            break
        if valid_crop(target / record["image"], record["sha256"]):
            good[record["sample_id"]] = record
    # This is our own resumable export journal, never a shared source file.
    temporary = journal.with_suffix(".recovering")
    with temporary.open("w") as stream:
        for record in good.values():
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    temporary.replace(journal)
    return good


def export_dataset(data_root):
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    from color_editing.clip_metrics import region_crop
    from color_editing.data import _load_mask, _mask_path, describe_sample, split_ids
    from color_editing.runs import save_json

    target = WORK / "data"
    target.mkdir(parents=True, exist_ok=True)
    recipe_path = target / "recipe.json"
    current = recipe()
    if recipe_path.exists() and read_json(recipe_path) != current:
        raise SystemExit("Color export recipe changed. Keep the existing export intact and inspect it first.")
    save_json(recipe_path, current)
    journal = target / "records.partial.jsonl"
    records = recover_records(journal, target)
    partitions = split_ids(35)
    completed_manifest = target / "dataset.json"
    if completed_manifest.exists():
        previous = read_json(completed_manifest)
        if previous.get("recipe") == current and len(records) == len(previous.get("records", [])):
            print(f"Real-region dataset already complete: {len(records):,} verified crops.", flush=True)
            return previous

    workers = max(1, min(4, len(os.sched_getaffinity(0)), int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    print(f"Export workers: {workers}; each photograph is decoded once for both regions.", flush=True)
    ordered = []
    with journal.open("a") as log, ThreadPoolExecutor(max_workers=workers) as pool:
        for split, ids in partitions.items():
            def prepare_image(image_id):
                image, image_records = None, []
                for region in ("hair", "cloth"):
                    sample_id = f"{image_id:05d}-{region}"
                    if sample_id in records:
                        record = records[sample_id]
                        if record["split"] != split:
                            raise RuntimeError(f"Unexpected split for {sample_id}")
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
                        label = canonical(stats["source_color"])
                        if label not in CLASSES:
                            raise RuntimeError(f"Unknown weak label: {label}")
                        relative = f"crops/{split}/{sample_id}.jpg"
                        image_hash = save_image(region_crop(image, mask, size=SIZE), target / relative)
                        record = {"sample_id": sample_id, "image_id": image_id, "region": region,
                                  "split": split, "label": label, "weak_label": stats["source_color"],
                                  "image": relative, "mask_fraction": fraction, "sha256": image_hash}
                    image_records.append(record)
                return image_records

            progress = tqdm(total=len(ids), desc=f"Export {split} real images", unit="image",
                            dynamic_ncols=True, mininterval=1.0)
            count = 0
            # Bounded submission avoids keeping 30,000 futures and decoded photos in memory.
            for start in range(0, len(ids), 64):
                for image_records in pool.map(prepare_image, ids[start:start + 64]):
                    for record in image_records:
                        if record["sample_id"] not in records:
                            records[record["sample_id"]] = record
                            log.write(json.dumps(record, allow_nan=False) + "\n")
                            log.flush()
                        ordered.append(record)
                        count += 1
                    progress.update(1)
                    progress.set_postfix(regions=count, refresh=False)
            progress.close()

    training = [record for record in ordered if record["split"] == "train"]
    if len(training) != 37971:
        raise RuntimeError(f"Expected 37,971 qualifying real training regions; found {len(training):,}.")
    counts = {split: dict(Counter(record["label"] for record in ordered if record["split"] == split))
              for split in partitions}
    manifest = {**current, "status": "completed", "recipe": current,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": "CelebAMask-HQ", "data_root": str(data_root),
                "split_image_counts": {key: len(value) for key, value in partitions.items()},
                "class_counts": counts, "records": ordered,
                "limitations": ["Labels are automatic nearest-median-Lab weak labels, not human ground truth.",
                                "Image IDs are disjoint across splits; identities are not guaranteed disjoint.",
                                "blond and yellow share a prototype and are merged into yellow for the CNN."]}
    save_json(completed_manifest, manifest)
    (target / "dataset.sha256").write_text(digest(completed_manifest) + "  dataset.json\n")
    print(f"Real-region export complete: {len(ordered):,} crops; {len(training):,} training regions.", flush=True)
    print("Class counts:", json.dumps(counts, sort_keys=True), flush=True)
    return manifest


def export_catalog(shared):
    from color_editing.runs import save_json
    paths = list((shared / "runs/wsun8").glob("*_evaluate_*"))
    followup = read_json(ROOT / "results/final-followup/evaluation.json")
    for name in ("nored", "p75"):
        group = followup["groups"][name]
        if not group.get("completed"):
            raise RuntimeError(f"Complete {name} evaluation before exporting its color-evaluation catalog.")
        paths.append(Path(group["run"]))
    runs = []
    seen = set()
    for path in sorted(paths):
        path = path.resolve()
        if str(path) in seen:
            continue
        seen.add(str(path))
        metadata = read_json(path / "run.json")
        if metadata.get("status") != "completed":
            print(f"Skipping incomplete evaluation: {path.name}", flush=True)
            continue
        methods = sorted(item.name for item in (path / "images").iterdir() if item.is_dir())
        cases = read_json(path / "cases.json")
        if not methods or not cases or not (path / "clip_summary.json").is_file():
            raise RuntimeError(f"Incomplete evaluation inputs: {path}")
        runs.append({"run_id": path.name, "path": str(path), "methods": methods,
                     "cases": len(cases), "cases_sha256": digest(path / "cases.json")})
    if len(runs) < 13:
        raise RuntimeError(f"Expected at least 11 shared and 2 personal evaluations, found {len(runs)}.")
    catalog = {"schema_version": SCHEMA, "runs": runs,
               "note": "All methods retained with run IDs; repeated baselines must not be counted as independent runs."}
    save_json(WORK / "data/evaluation_catalog.json", catalog)
    print(f"Evaluation catalog: {len(runs)} runs, {sum(len(run['methods']) for run in runs)} method/run groups.", flush=True)
    return catalog


def export_audit(data_root, catalog):
    from PIL import Image
    from tqdm import tqdm
    from color_editing.clip_metrics import region_crop
    from color_editing.data import load_sample, describe_sample
    from color_editing.evaluate import case_name
    from color_editing.runs import save_json

    def find_run(suffix):
        matches = [run for run in catalog["runs"] if run["run_id"].endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one evaluation ending {suffix}: {len(matches)} found")
        return Path(matches[0]["path"])

    reference = find_run("_b9510685")
    followup = read_json(ROOT / "results/final-followup/evaluation.json")["groups"]
    method_runs = [("B0", "B0", reference), ("B1", "B1", reference), ("Main", "Mainv", reference),
                   ("NoRed", "NoRed-r35", Path(followup["nored"]["run"])),
                   ("P75", "P75-r35", Path(followup["p75"]["run"]))]
    source_labels = read_json(reference / "clip_summary.json")["labels"]
    source_index = {(row["image_id"], row["region"]): row for row in source_labels}
    rng = random.Random(35)
    selections = []
    for region in ("hair", "cloth"):
        for row in rng.sample([row for row in source_labels if row["region"] == region], 50):
            selections.append({"kind": "source", "region": region, "source_image_id": row["image_id"],
                               "lab_color": canonical(row["rule_label"]), "weak_label": row["rule_label"],
                               "clip_color": canonical(row["clip_label"]), "run_id": reference.name,
                               "sample_id": f"{row['image_id']:05d}-{region}"})
    # Source and edited checks are both required; the public items never expose their kind.
    rng.shuffle(selections)
    method_runs.insert(2, ("E1", "E1v", reference))
    method_runs.insert(4, ("Light", "Light", find_run("_eeecddc1")))
    reference_cases = read_json(reference / "cases.json")
    selected_cases = []
    for region, targets in (("hair", ("blond", "red", "blue")), ("cloth", ("red", "blue", "white"))):
        for target in targets:
            pool = [case for case in reference_cases if case["region"] == region and case["target_color"] == target]
            selected_cases.extend(rng.sample(pool, 2))
    edited = []
    for method, folder, run in method_runs:
        predictions = [json.loads(line) for line in (run / "clip_metrics.jsonl").read_text().splitlines()]
        index = {(row["image_id"], row["region"], row["target_color"]): row
                 for row in predictions if row["method"] == folder}
        cases = read_json(run / "cases.json")
        if cases != reference_cases:
            raise RuntimeError(f"Audit requires exactly paired test cases: {run}")
        for case in selected_cases:
            region, target = case["region"], case["target_color"]
            key = (case["image_id"], region, target)
            original = source_index[(case["image_id"], region)]
            edited.append({"kind": "edited", "region": region, "source_image_id": case["image_id"],
                           "method": folder, "display_method": method, "method_folder": folder,
                           "target_color": canonical(target), "target_color_original": target,
                           "clip_color": canonical(index[key]["clip_color"]),
                           "source_lab_color": canonical(original["rule_label"]),
                           "source_clip_color": canonical(original["clip_label"]),
                           "run_id": run.name, "run_path": str(run), "case_name": case_name(case)})
    rng.shuffle(edited)
    selections.extend(edited)
    private_items, public_items = [], []
    pack = WORK / "human-audit/pack"
    key_path = WORK / "human-audit/key.json"
    if key_path.exists():
        previous = read_json(key_path)
        old_items = previous.get("items", [])
        if len(old_items) != len(selections):
            raise RuntimeError("Audit selection changed; keep the existing annotator pack intact.")
        for number, (old, new) in enumerate(zip(old_items, selections), 1):
            if old.get("item_id") != f"{number:04d}" or any(old.get(key) != value for key, value in new.items()):
                raise RuntimeError("Audit selection changed; keep the existing annotator pack intact.")
            if not valid_crop(pack / f"images/{number:04d}.jpg", old["image_sha256"], size=256):
                raise RuntimeError(f"Audit image {number:04d} is missing or changed. Preserve any existing labels before repair.")
        if read_json(pack / "manifest.json")["pack_id"] != previous["pack_id"]:
            raise RuntimeError("Audit manifest and private key do not match.")
        print(f"Blinded audit already complete: {len(selections)} verified items.", flush=True)
        return
    for number, selection in enumerate(tqdm(selections, desc="Export blinded audit", unit="item", dynamic_ncols=True), 1):
        item_id = f"{number:04d}"
        sample = load_sample(data_root, selection["source_image_id"], selection["region"], dilation=0)
        if selection["kind"] == "edited":
            with Image.open(Path(selection["run_path"]) / "images" / selection["method_folder"] /
                            (selection["case_name"] + ".png")) as image:
                sample["image"] = image.convert("RGB")
            selection["lab_color"] = canonical(describe_sample(sample)["source_color"])
        image_path = f"images/{item_id}.jpg"
        image_hash = save_image(region_crop(sample["image"], sample["mask"], size=256), pack / image_path)
        public_items.append({"item_id": item_id, "image": image_path, "region": selection["region"]})
        private_items.append({"item_id": item_id, **selection, "image_sha256": image_hash})
    pack_id = "group35-color-audit-" + hashlib.sha256(
        json.dumps(private_items, sort_keys=True).encode()).hexdigest()[:16]
    public = {"pack_id": pack_id, "items": public_items}
    private = {"pack_id": pack_id, "items": private_items, "audit_crop_size": 256,
               "sampling": {"seed": 35, "source_items": 100, "edited_items": 84,
                            "source_selection": "Uniform random 50 hair + 50 clothing from the 200 evaluated source regions",
                            "edited_selection": "7 methods x 2 regions x 3 targets x 2 paired cases; seed-35 adapters only",
                            "limitation": "Small selected test subset; clothing pattern is human-labeled. Edited cases are paired across methods."},
               "source_item_ids": [item["item_id"] for item in private_items if item["kind"] == "source"]}
    save_json(pack / "manifest.json", public)
    save_json(key_path, private)
    print(f"Blinded audit ready: {len(public_items)} items (first 100 source checks; last 84 edited checks).", flush=True)
    print(f"Give annotators only: {pack}", flush=True)
    print(f"Keep this answer key private: {key_path}", flush=True)


def main():
    print("Starting color-data preparation; imports and data checks follow.", flush=True)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", "dataset", "audit"), default="all")
    args = parser.parse_args()
    if Path("/ocean/projects/cis260330p").exists():
        if not os.environ.get("SLURM_JOB_ID"):
            raise SystemExit("Run this image-processing export inside an allocated compute-node terminal.")
        if os.environ.get("SLURM_JOB_ACCOUNT") != "cis260330p":
            raise SystemExit("Use the project allocation: interact -A cis260330p ...")
    sys.path.insert(0, str(ROOT / "src"))
    from color_editing.runs import save_json
    shared = Path(read_json(ROOT / "configs/main.json")["shared_root"])
    data_root = shared / "data/CelebAMask-HQ"
    print("Color-data export: CPU image processing only; shared sources stay read only.", flush=True)
    with export_lock():
        if args.only in ("all", "dataset"):
            export_dataset(data_root)
        catalog = export_catalog(shared)
        if args.only in ("all", "audit"):
            export_audit(data_root, catalog)
        save_json(WORK / "export-status.json", {"status": "completed", "selection": args.only,
                  "finished_utc": datetime.now(timezone.utc).isoformat()})
    print("Color-data preparation complete.", flush=True)


if __name__ == "__main__":
    main()
