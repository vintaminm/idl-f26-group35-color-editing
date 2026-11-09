"""Download evaluation weights to this clone; no GPU or inference is used.

The user runs this after installing requirements-evaluation.txt:
    .venv/bin/python scripts/final/prepare_evaluation.py
"""

from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
HF_HOME = RESULTS / "_hf_cache"
TORCH_HOME = RESULTS / "_torch_cache"
INSIGHTFACE_ROOT = RESULTS / "_insightface_cache"
ASSETS = RESULTS / "final-followup/evaluation-assets.json"
CLIP_ID = "openai/clip-vit-large-patch14"
WEIGHTS = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
    "inception": "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/"
                 "weights-inception-2015-12-05-6726825d.pth",
}


def describe(path):
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        raise RuntimeError(f"Missing or empty weight file: {path}")
    size = path.stat().st_size
    large = size >= 64 * 1024 * 1024
    processed = 0
    last_update = time.monotonic()
    if large:
        print(f"Hashing {path.name}: {size / 1024 ** 2:.1f} MiB", flush=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            processed += len(chunk)
            if large and time.monotonic() - last_update >= 30:
                print(f"Hashing {path.name}: {processed / 1024 ** 2:.1f}/{size / 1024 ** 2:.1f} MiB",
                      flush=True)
                last_update = time.monotonic()
    if large:
        print(f"Hash complete: {path.name}", flush=True)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def check_environment():
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise SystemExit("Use this repository's .venv/bin/python.")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    os.environ.update(HF_HOME=str(HF_HOME), TORCH_HOME=str(TORCH_HOME),
                      HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0",
                      HF_HUB_DISABLE_TELEMETRY="1", PYTHONNOUSERSITE="1")
    packages = {}
    for line in (ROOT / "requirements-evaluation.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            raise SystemExit(f"Missing {name}. Install requirements-evaluation.txt first.")
        if installed != version:
            raise SystemExit(f"{name}: expected {version}, found {installed}. "
                             "Install requirements-evaluation.txt with the PSC constraints first.")
        packages[name] = installed
    # Catch missing transitive dependencies before any large download.
    for module in ("torch", "torchvision", "lpips", "matplotlib", "torchmetrics",
                   "torch_fidelity", "insightface", "onnxruntime", "scipy", "cv2", "skimage"):
        print(f"Checking import: {module}", flush=True)
        importlib.import_module(module)
    print("Checking CLIP and Inception interfaces.", flush=True)
    from transformers import CLIPModel, CLIPProcessor
    from torchmetrics.image.fid import NoTrainInceptionV3
    import onnxruntime
    if "CPUExecutionProvider" not in onnxruntime.get_available_providers():
        raise SystemExit("ONNX Runtime CPUExecutionProvider is unavailable.")
    for name in ("torch", "torchvision", "numpy", "transformers", "huggingface_hub",
                 "onnx", "opencv-python", "scikit-image"):
        packages[name] = metadata.version(name)
    return packages, onnxruntime.get_available_providers()


def main():
    print("Checking evaluation environment and package imports; no model inference is run.", flush=True)
    packages, providers = check_environment()
    ASSETS.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with ASSETS.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another evaluation asset preparation is running.")
        from huggingface_hub import snapshot_download
        from insightface.utils.storage import ensure_available
        from torch.hub import download_url_to_file
        from color_editing.runs import save_json

        save_json(ASSETS, {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat()})
        print("[1/3] CLIP ViT-L/14: download safetensors and processor files.", flush=True)
        snapshot = Path(snapshot_download(
            repo_id=CLIP_ID, revision="main", cache_dir=str(HF_HOME / "hub"),
            allow_patterns=["model.safetensors", "*.json", "*.txt"],
        ))
        required = ("model.safetensors", "config.json", "preprocessor_config.json",
                    "tokenizer_config.json", "vocab.json", "merges.txt")
        for name in required:
            if not (snapshot / name).is_file():
                raise RuntimeError(f"CLIP snapshot is incomplete: {snapshot / name}")
        print("CLIP downloaded; recording file hashes.", flush=True)
        clip_files = {str(p.relative_to(snapshot)): describe(p)
                      for p in sorted(snapshot.rglob("*")) if p.is_file()}

        print("[2/3] AlexNet and Inception-v3 weight files.", flush=True)
        checkpoints = TORCH_HOME / "hub/checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        torch_files = {}
        for name, url in WEIGHTS.items():
            target = checkpoints / url.rsplit("/", 1)[1]
            if not target.is_file() or not target.stat().st_size:
                temporary = target.with_name(target.name + ".partial")
                print(f"Downloading {name}: {target}", flush=True)
                download_url_to_file(url, str(temporary), progress=True)
                temporary.replace(target)
            else:
                print(f"Reuse {name}: {target}", flush=True)
            torch_files[name] = {**describe(target), "url": url}

        print("[3/3] ArcFace buffalo_l: download/unpack only; no inference session.", flush=True)
        pack = Path(ensure_available("models", "buffalo_l", root=str(INSIGHTFACE_ROOT)))
        for name in ("det_10g.onnx", "w600k_r50.onnx"):
            if not (pack / name).is_file() or not (pack / name).stat().st_size:
                raise RuntimeError(f"ArcFace model pack is incomplete: {pack / name}. "
                                   "Inspect the incomplete model directory before retrying.")
        arc_files = {p.name: describe(p) for p in sorted(pack.glob("*.onnx"))}
        save_json(ASSETS, {
            "status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
            "caches": {"hf_home": str(HF_HOME), "torch_home": str(TORCH_HOME),
                       "insightface_root": str(INSIGHTFACE_ROOT)},
            "clip": {"repo_id": CLIP_ID, "requested_revision": "main",
                     "resolved_revision": snapshot.name, "path": str(snapshot), "files": clip_files},
            "torch": torch_files,
            "arcface": {"root": str(INSIGHTFACE_ROOT), "path": str(pack), "files": arc_files},
            "packages": packages, "onnxruntime_providers": providers,
        })
        print(f"\nEvaluation weights ready. Manifest: {ASSETS}", flush=True)
        print("No model inference or training was run.", flush=True)


if __name__ == "__main__":
    main()
