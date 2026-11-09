"""A small, independently trained color evaluator, not a human ground truth.

Only exported real training crops and their weak Lab labels train this model.
Validation chooses a checkpoint; test crops and generated edits never train it.
There are no pretrained weights, network requests, or package installations here.
"""

import argparse
from collections import Counter, defaultdict
import contextlib
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import _load_mask, _mask_path, split_ids
from .clip_metrics import region_crop


CLASSES = ["black", "brown", "yellow", "red", "blue", "green", "purple", "pink", "white", "gray"]
ARCHITECTURE = "region-color-cnn-v1"


def canonical(color):
    return {"blond": "yellow", "grey": "gray"}.get(color, color)


def display_color(color, region):
    return "blond" if region == "hair" and color == "yellow" else color


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty result table")
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def git_provenance():
    root = Path(__file__).resolve().parents[2]
    result = {}
    for key, command in (("commit", ["rev-parse", "HEAD"]),
                         ("branch", ["branch", "--show-current"]),
                         ("working_tree", ["status", "--porcelain"])):
        try:
            result[key] = subprocess.check_output(["git", "-C", str(root), *command],
                                                  text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError) as error:
            result[key] = f"Unavailable: {type(error).__name__}"
    return result


def execution_provenance():
    return {"utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"), "slurm_account": os.getenv("SLURM_JOB_ACCOUNT")}


def allocation_guard(args):
    if Path("/ocean/projects/cis260330p").exists():
        if not os.getenv("SLURM_JOB_ID") or os.getenv("SLURM_JOB_ACCOUNT") != "cis260330p":
            raise SystemExit("On PSC, run CNN commands inside a compute allocation requested with -A cis260330p.")
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK", "")
    if slurm_cpus:
        try:
            available = min(available, int(slurm_cpus.split("(")[0]))
        except ValueError:
            raise SystemExit(f"Cannot interpret allocated SLURM_CPUS_PER_TASK={slurm_cpus!r}")
    args.threads = max(1, min(args.threads, available))
    args.workers = min(args.workers, max(0, args.threads - 1))
    os.environ.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                      OPENBLAS_NUM_THREADS=str(args.threads))
    print(f"CPU budget: {args.threads} threads, {args.workers} data workers", flush=True)


def plot_training(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read_json(output / "history.json")
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), constrained_layout=True)
    axes[0].plot(epochs, [r["train_loss"] for r in rows], label="Train (class-weighted)")
    axes[0].plot(epochs, [r["val_loss"] for r in rows], label="Validation (unweighted)")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].legend(fontsize=8)
    axes[1].plot(epochs, [r["val_accuracy"] for r in rows], label="Accuracy")
    axes[1].plot(epochs, [r["val_macro_accuracy"] for r in rows], label="Macro recall (selection)")
    axes[1].set(xlabel="Epoch", ylabel="Agreement with validation Lab weak labels", ylim=(0, 1))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"training_curves.{suffix}", dpi=160)
    plt.close(fig)


def plot_confusion(output, result, supported):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrix = np.asarray(result["confusion_matrix"])
    n = matrix.sum(axis=1, keepdims=True)
    proportions = np.divide(matrix, n, out=np.zeros_like(matrix, dtype=float), where=n > 0)
    labels = [c if c in supported else c + " *" for c in CLASSES]
    fig, ax = plt.subplots(figsize=(7.3, 6.2), constrained_layout=True)
    image = ax.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.045, label="Fraction within each weak-label class")
    ax.set(xticks=range(len(CLASSES)), yticks=range(len(CLASSES)),
           xticklabels=labels, yticklabels=[f"{c} (n={int(total)})" for c, total in zip(labels, n[:, 0])],
           xlabel="CNN prediction", ylabel="Held-out real-region Lab weak label",
           title="CNN test confusion: weak-label agreement, not human accuracy\n* = no training examples")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for row in range(len(CLASSES)):
        for col in range(len(CLASSES)):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=7,
                    color="white" if proportions[row, col] > 0.55 else "black")
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"test_confusion.{suffix}", dpi=160)
    plt.close(fig)


@contextlib.contextmanager
def single_writer(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".cnn.lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another CNN command is using this output directory.")
        yield


def read_dataset(path):
    path = Path(path).resolve()
    data = read_json(path)
    if data.get("status") != "completed" or data.get("schema_version") != 1:
        raise ValueError("The crop export must be completed with schema_version 1")
    if data.get("crop_size") != 128 or data.get("classes") != CLASSES:
        raise ValueError("Expected 128-pixel crops and the fixed canonical 10-color palette")
    records = data.get("records", [])
    if not records:
        raise ValueError("Empty crop dataset")
    partitions = {k: set(v) for k, v in split_ids(data.get("split_seed", 35)).items()}
    seen = set()
    for r in records:
        if r["sample_id"] in seen:
            raise ValueError(f"Duplicate crop: {r['sample_id']}")
        seen.add(r["sample_id"])
        if r["region"] not in {"hair", "cloth"} or r["label"] not in CLASSES:
            raise ValueError(f"Invalid region/color in {r['sample_id']}")
        if r["split"] not in partitions or r["image_id"] not in partitions[r["split"]]:
            raise ValueError(f"Image ID crosses the fixed original-photo split: {r['sample_id']}")
        image = (path.parent / r["image"]).resolve()
        if not image.is_relative_to(path.parent) or not image.is_file():
            raise ValueError(f"Missing or unsafe crop path: {image}")
    if {r["split"] for r in records} != {"train", "val", "test"}:
        raise ValueError("Need nonempty train, val, and test splits")
    return path, data, records


def image_tensor(image):
    # No ImageNet normalization or color-changing transforms: these destroy the task signal.
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0


class CropDataset(Dataset):
    def __init__(self, records, directory, training=False):
        self.records = records
        self.directory = directory
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        with Image.open(self.directory / r["image"]) as source:
            image = source.convert("RGB")
            if image.size != (128, 128):
                raise ValueError(f"Unexpected crop dimensions: {r['image']}")
            if self.training and torch.rand(()).item() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            result = image_tensor(image)
        return result, CLASSES.index(r["label"]), index


class ColorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        before = 3
        for channels in (24, 48, 96, 128):
            blocks.extend([
                nn.Conv2d(before, channels, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, channels), nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, channels), nn.SiLU(),
            ])
            before = channels
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((2, 2)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512, 128), nn.SiLU(),
                                  nn.Dropout(0.1), nn.Linear(128, len(CLASSES)))

    def forward(self, images):
        return self.head(self.features(images))


def device_for(name, threads):
    torch.set_num_threads(threads)
    device = torch.device("cuda" if name == "auto" and torch.cuda.is_available()
                          else "cpu" if name == "auto" else name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; activate the project venv on the allocated node")
    print(f"CNN device: {device}; CPU threads: {threads}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}; Slurm job: {os.getenv('SLURM_JOB_ID', 'not set')}", flush=True)
    return device


def metrics(labels, predictions):
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for label, pred in zip(labels, predictions):
        matrix[label, pred] += 1
    per_class = []
    for i, color in enumerate(CLASSES):
        n = int(matrix[i].sum())
        predicted = int(matrix[:, i].sum())
        per_class.append({"color": color, "n": n, "correct": int(matrix[i, i]),
                          "recall": float(matrix[i, i] / n) if n else None,
                          "precision": float(matrix[i, i] / predicted) if predicted else None})
    recalls = [r["recall"] for r in per_class if r["n"]]
    return {"n": len(labels), "accuracy": float(np.mean(np.asarray(labels) == predictions)),
            "macro_accuracy": float(np.mean(recalls)), "classes": CLASSES,
            "per_class": per_class, "confusion_matrix": matrix.tolist(),
            "warning": "Agreement with automatic Lab weak labels; not human-verified accuracy."}


def supported_logits(logits, supported):
    if not supported:
        raise ValueError("Classifier has no supported training classes")
    valid = torch.zeros(len(CLASSES), dtype=torch.bool, device=logits.device)
    valid[[CLASSES.index(c) for c in supported]] = True
    return logits.masked_fill(~valid[None], -torch.inf)


def loaders(records, directory, args, generator):
    result = {}
    for split in ("train", "val", "test"):
        picked = [r for r in records if r["split"] == split]
        dataset = CropDataset(picked, directory, training=split == "train")
        result[split] = DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train",
                                   num_workers=args.workers, pin_memory=False, generator=generator,
                                   persistent_workers=False)
    return result


def validate(model, loader, device, supported, description):
    model.eval()
    predictions, labels, confidences = [], [], []
    loss_sum = 0.0
    with torch.inference_mode():
        for images, targets, _ in tqdm(loader, desc=description, unit="batch", dynamic_ncols=True):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss_sum += float(nn.functional.cross_entropy(logits, targets, reduction="sum"))
            probabilities = supported_logits(logits, supported).softmax(dim=1)
            conf, pred = probabilities.max(dim=1)
            predictions.extend(pred.cpu().tolist())
            labels.extend(targets.cpu().tolist())
            confidences.extend(conf.cpu().tolist())
    result = metrics(labels, predictions)
    result["loss"] = loss_sum / len(labels)
    result["supported_classes"] = supported
    result["unsupported_classes"] = [c for c in CLASSES if c not in supported]
    for row in result["per_class"]:
        row["training_supported"] = row["color"] in supported
    return result, predictions, confidences


def training_recipe(args, manifest):
    return {"architecture": ARCHITECTURE, "classes": CLASSES, "crop_size": 128,
            "manifest_sha256": sha256(manifest), "code_sha256": sha256(__file__),
            "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "patience": args.patience, "augmentation": "horizontal flip only",
            "selection": "maximum validation macro recall, absent validation classes excluded",
            "supervision": "real training crops with automatic Lab weak labels; no synthetic examples",
            "loss": "cross entropy with clipped inverse-square-root class-frequency weights"}


def train(args):
    manifest, data, records = read_dataset(args.manifest)
    output = args.output.resolve()
    recipe = training_recipe(args, manifest)
    recipe_id = fingerprint(recipe)
    with single_writer(output):
        state_path = output / "training.json"
        old = read_json(state_path) if state_path.exists() else {}
        if old and old.get("recipe_id") != recipe_id:
            raise SystemExit("CNN training inputs/settings/code changed. Inspect existing outputs or choose a new --output.")
        if old.get("status") == "completed":
            if sha256(output / "best.pt") != old["best_sha256"]:
                raise SystemExit("Completed CNN checkpoint changed; refusing to silently reuse it.")
            print(f"Reuse completed CNN: {output / 'best.pt'}", flush=True)
            plot_training(output)
            return
        device = device_for(args.device, args.threads)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        generator = torch.Generator().manual_seed(args.seed)
        count = Counter(r["label"] for r in records if r["split"] == "train")
        supported = [c for c in CLASSES if count[c] > 0]
        missing = [c for c in CLASSES if count[c] == 0]
        print("Training class counts:", dict(count), flush=True)
        if missing:
            print("WARNING: absent training colors:", missing,
                  "; these cannot be treated as a supported CNN evaluation.", flush=True)
        if len(supported) < 2:
            raise ValueError("Need at least two supported training colors")
        median = float(np.median([count[c] for c in supported]))
        weights = torch.tensor([min(5.0, max(0.2, (median / count[c]) ** 0.5))
                                if count[c] else 0.0 for c in CLASSES], device=device)
        weights = weights / weights[weights > 0].mean()
        model = ColorCNN().to(device)
        print(f"CNN parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        start_epoch, best, stale, history = 0, -1.0, 0, []
        latest = output / "latest.pt"
        if latest.exists():
            checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
            if checkpoint["recipe_id"] != recipe_id:
                raise ValueError("Interrupted CNN checkpoint has different inputs")
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch, best, stale = checkpoint["epoch"], checkpoint["best"], checkpoint["stale"]
            history = checkpoint["history"]
            generator.set_state(checkpoint["loader_rng"])
            torch.set_rng_state(checkpoint["torch_rng"])
            if device.type == "cuda" and checkpoint.get("cuda_rng"):
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
            print(f"Resume after epoch {start_epoch}", flush=True)
        all_loaders = loaders(records, manifest.parent, args, generator)
        state = {"status": "running", "recipe_id": recipe_id, "recipe": recipe,
                 "manifest": str(manifest), "supported_classes": supported, "missing_classes": missing,
                 "training_class_counts": {c: count[c] for c in CLASSES},
                 "split_counts": dict(Counter(r["split"] for r in records)),
                 "trainable_parameters": sum(p.numel() for p in model.parameters()),
                 "torch": str(torch.__version__), "device": str(device), "label_source": data.get("label_source"),
                 "git": git_provenance(), "started": old.get("started", execution_provenance()),
                 "last_execution": execution_provenance()}
        write_json(state_path, state)
        started = time.monotonic()
        for epoch in range(start_epoch, args.epochs):
            if stale >= args.patience:
                break
            model.train()
            total_loss, examples = 0.0, 0
            progress = tqdm(all_loaders["train"], desc=f"CNN epoch {epoch + 1}/{args.epochs}",
                            unit="batch", dynamic_ncols=True, mininterval=1)
            for images, targets, _ in progress:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, targets, weight=weights)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite CNN training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.detach()) * len(targets)
                examples += len(targets)
                progress.set_postfix(loss=f"{total_loss / examples:.4f}", refresh=False)
            result, _, _ = validate(model, all_loaders["val"], device, supported, "CNN validation")
            scheduler.step()
            row = {"epoch": epoch + 1, "train_loss": total_loss / examples,
                   "val_loss": result["loss"], "val_accuracy": result["accuracy"],
                   "val_macro_accuracy": result["macro_accuracy"], "learning_rate": optimizer.param_groups[0]["lr"]}
            history.append(row)
            improved = result["macro_accuracy"] > best + 1e-4
            if improved:
                best, stale = result["macro_accuracy"], 0
                save_checkpoint(output / "best.pt", {"architecture": ARCHITECTURE, "classes": CLASSES,
                    "crop_size": 128, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "recipe_id": recipe_id, "recipe": recipe, "supported_classes": supported, "epoch": epoch + 1})
                write_json(output / "validation.json", result)
            else:
                stale += 1
            write_json(output / "history.json", history)
            write_csv(output / "history.csv", history)
            save_checkpoint(latest, {"recipe_id": recipe_id, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1, "best": best, "stale": stale, "history": history,
                "loader_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
            state.update(epoch=epoch + 1, best_val_macro_accuracy=best, stale_epochs=stale)
            write_json(state_path, state)
            print(f"Epoch {epoch + 1}: train loss {row['train_loss']:.4f}; "
                  f"val accuracy {result['accuracy']:.3f}, macro {result['macro_accuracy']:.3f}; "
                  f"best {best:.3f}; elapsed {(time.monotonic() - started) / 60:.1f} min", flush=True)
            if stale >= args.patience:
                print(f"Early stop: no macro-validation improvement for {args.patience} epochs", flush=True)
                break
        plot_training(output)
        state.update(status="completed", best_sha256=sha256(output / "best.pt"),
                     elapsed_seconds=time.monotonic() - started, finished=execution_provenance())
        write_json(state_path, state)
        print(f"CNN training completed: {output / 'best.pt'}", flush=True)


def load_model(output, device, manifest):
    state = read_json(output / "training.json")
    if state.get("status") != "completed":
        raise ValueError("CNN training has not completed")
    if state["recipe"]["manifest_sha256"] != sha256(manifest):
        raise ValueError("Crop manifest differs from the classifier's training manifest")
    model_sha = sha256(output / "best.pt")
    if model_sha != state["best_sha256"]:
        raise ValueError("CNN best checkpoint changed")
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["architecture"] != ARCHITECTURE or checkpoint["classes"] != CLASSES:
        raise ValueError("Unsupported CNN checkpoint architecture/palette")
    model = ColorCNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["supported_classes"], model_sha


def source_predictions(model, supported, device, manifest, records, output, args, model_sha):
    expected = {"model_sha256": model_sha, "manifest_sha256": sha256(manifest)}
    state_path = output / "source_evaluation.json"
    if state_path.exists():
        state = read_json(state_path)
        if state.get("inputs") != expected:
            raise ValueError("Existing source CNN predictions have different inputs")
        if state.get("status") == "completed" and (output / "source_predictions.jsonl").is_file():
            if sha256(output / "source_predictions.jsonl") != state["predictions_sha256"]:
                raise ValueError("Saved source CNN predictions changed")
            print("Reuse held-out real-source CNN predictions", flush=True)
            plot_confusion(output, read_json(output / "test_metrics.json"), supported)
            return
    picked = [r for r in records if r["split"] == "test"]
    loader = DataLoader(CropDataset(picked, manifest.parent), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False)
    result, predicted, conf = validate(model, loader, device, supported, "CNN held-out real test")
    rows = [{"sample_id": r["sample_id"], "image_id": r["image_id"], "region": r["region"],
             "weak_label": r.get("weak_label", r["label"]), "canonical_label": r["label"],
             "cnn_color": display_color(CLASSES[p], r["region"]), "confidence": confidence,
             "cnn_correct_weak_label": CLASSES[p] == r["label"],
             "label_supported": r["label"] in supported}
            for r, p, confidence in zip(picked, predicted, conf)]
    write_jsonl(output / "source_predictions.jsonl", rows)
    write_json(output / "test_metrics.json", result)
    plot_confusion(output, result, supported)
    state = {"status": "completed", "inputs": expected,
             "predictions_sha256": sha256(output / "source_predictions.jsonl"), "n": len(rows)}
    write_json(state_path, state)
    print(f"Held-out real-source Lab agreement: {result['accuracy']:.3f}, "
          f"macro {result['macro_accuracy']:.3f} (weak labels, not human accuracy)", flush=True)


def run_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        for target in ("all", row["target_color"]):
            groups[(row["method"], row["region"], target)].append(row)
    result = []
    for (method, region, target), items in sorted(groups.items()):
        supported = [r for r in items if r["target_supported"]]
        result.append({"method": method, "region": region, "target": target, "n": len(items),
                       "cnn_color_accuracy": sum(r["cnn_correct"] for r in items) / len(items),
                       "supported_n": len(supported), "unsupported_n": len(items) - len(supported),
                       "supported_color_accuracy": (sum(r["cnn_correct"] for r in supported) / len(supported)
                                                    if supported else None),
                       "mean_confidence": sum(r["confidence"] for r in items) / len(items)})
    return result


def evaluate_run(model, supported, device, entry, output, args, model_sha, mask_cache):
    run = Path(entry["path"]).resolve()
    run_id = entry.get("run_id", run.name)
    if Path(run_id).name != run_id:
        raise ValueError("Catalog run_id must be a single directory name")
    directory = output / "evaluations" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    config = read_json(run / "config.json")
    cases = read_json(run / "cases.json")
    methods = entry["methods"]
    if len(methods) != len(set(methods)) or not methods or not cases:
        raise ValueError(f"Empty or duplicate evaluation methods/cases in {run}")
    data_root = Path(config["shared_root"]) / "data/CelebAMask-HQ"
    signatures = []
    for method in methods:
        if Path(method).name != method:
            raise ValueError(f"Unsafe method name: {method}")
        for case in cases:
            path = run / "images" / method / f"{case['image_id']:05d}-{case['region']}-{case['target_color']}.png"
            info = path.stat()
            signatures.append([method, path.name, info.st_size, info.st_mtime_ns])
    inputs = {"model_sha256": model_sha, "cases_sha256": sha256(run / "cases.json"),
              "config_sha256": sha256(run / "config.json"), "methods": methods,
              "generated_file_stats_sha256": fingerprint(signatures), "run": str(run),
              "crop": "raw mask, existing CLIP region_crop at 128 pixels; gray outside"}
    state_path = directory / "run.json"
    if state_path.exists():
        state = read_json(state_path)
        if state.get("inputs") != inputs:
            raise ValueError(f"Existing CNN evaluation inputs changed: {directory}")
        if state.get("status") == "completed":
            if sha256(directory / "cnn_metrics.jsonl") != state["metrics_sha256"]:
                raise ValueError(f"Saved CNN metrics changed: {directory}")
            print(f"Reuse completed CNN evaluation: {run_id}", flush=True)
            return {"run_id": run_id, "path": str(directory), "n": state["n"]}
    write_json(state_path, {"status": "running", "inputs": inputs})
    rows = []
    for method in tqdm(methods, desc=f"CNN {run_id[-20:]}", unit="method", dynamic_ncols=True):
        for start in tqdm(range(0, len(cases), args.batch_size), desc=f"CNN {method}",
                          unit="batch", dynamic_ncols=True, mininterval=1):
            batch = cases[start:start + args.batch_size]
            tensors = []
            for case in batch:
                key = (str(data_root), case["image_id"], case["region"])
                if key not in mask_cache:
                    mask_cache[key] = _load_mask(_mask_path(data_root, case["image_id"], case["region"]))
                path = run / "images" / method / f"{case['image_id']:05d}-{case['region']}-{case['target_color']}.png"
                with Image.open(path) as raw:
                    if raw.size != (512, 512):
                        raise ValueError(f"Unexpected generated image dimensions: {path}")
                    crop = region_crop(raw.convert("RGB"), mask_cache[key], size=128)
                tensors.append(image_tensor(crop))
            with torch.inference_mode():
                logits = model(torch.stack(tensors).to(device))
                confidence, prediction = supported_logits(logits, supported).softmax(1).max(1)
            for case, index, conf in zip(batch, prediction.cpu().tolist(), confidence.cpu().tolist()):
                pred = CLASSES[index]
                rows.append({**case, "run_id": run_id, "method": method,
                             "cnn_color": display_color(pred, case["region"]),
                             "cnn_correct": pred == canonical(case["target_color"]),
                             "confidence": conf, "target_supported": canonical(case["target_color"]) in supported})
    table = run_summary(rows)
    write_jsonl(directory / "cnn_metrics.jsonl", rows)
    write_json(directory / "cnn_summary.json", table)
    write_csv(directory / "cnn_summary.csv", table)
    write_json(state_path, {"status": "completed", "inputs": inputs, "n": len(rows),
                           "metrics_sha256": sha256(directory / "cnn_metrics.jsonl"),
                           "warning": "Weak-label-trained CNN proxy; validate against held-out human annotations."})
    for row in table:
        if row["target"] == "all":
            print(f"{row['method']} {row['region']}: CNN accuracy {row['cnn_color_accuracy']:.3f}; "
                  f"n={row['n']}, unsupported targets={row['unsupported_n']}", flush=True)
    return {"run_id": run_id, "path": str(directory), "n": len(rows)}


def evaluate(args):
    manifest, _, records = read_dataset(args.manifest)
    output = args.output.resolve()
    with single_writer(output):
        device = device_for(args.device, args.threads)
        model, supported, model_sha = load_model(output, device, manifest)
        source_predictions(model, supported, device, manifest, records, output, args, model_sha)
        catalog = read_json(args.catalog)
        if catalog.get("schema_version") != 1 or not catalog.get("runs"):
            raise ValueError("Need a nonempty schema_version 1 evaluation catalog")
        ids = [r.get("run_id", Path(r["path"]).name) for r in catalog["runs"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate evaluation run IDs in catalog")
        entries, mask_cache = [], {}
        for entry in tqdm(catalog["runs"], desc="CNN evaluation runs", unit="run", dynamic_ncols=True):
            entries.append(evaluate_run(model, supported, device, entry, output, args, model_sha, mask_cache))
            write_json(output / "evaluation_index.json", {"status": "running", "model_sha256": model_sha,
                "catalog_sha256": sha256(args.catalog), "runs": entries})
        write_json(output / "evaluation_index.json", {"status": "completed", "model_sha256": model_sha,
            "catalog_sha256": sha256(args.catalog), "runs": entries, "source_predictions": "source_predictions.jsonl"})
        print(f"All CNN evaluation completed: {output / 'evaluation_index.json'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate"):
        p = commands.add_parser(command)
        p.add_argument("--manifest", type=Path, default=Path("results/final-color/data/dataset.json"))
        p.add_argument("--output", type=Path, default=Path("results/final-color/cnn"))
        p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--batch-size", type=int, default=128)
        if command == "train":
            p.add_argument("--seed", type=int, default=35)
            p.add_argument("--epochs", type=int, default=15)
            p.add_argument("--patience", type=int, default=4)
            p.add_argument("--learning-rate", type=float, default=1e-3)
            p.add_argument("--weight-decay", type=float, default=1e-4)
        else:
            p.add_argument("--catalog", type=Path, default=Path("results/final-color/data/evaluation_catalog.json"))
    args = parser.parse_args(argv)
    if args.threads < 1 or args.workers < 0 or args.batch_size < 1:
        parser.error("threads/batch size must be positive, workers nonnegative")
    if args.command == "train" and (args.epochs < 1 or args.patience < 1 or args.learning_rate <= 0):
        parser.error("epochs, patience, and learning rate must be positive")
    print(f"CNN {args.command}: starting; no package installation or external downloads", flush=True)
    allocation_guard(args)
    (train if args.command == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
