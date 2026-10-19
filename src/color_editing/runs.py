"""Explicit run bookkeeping; importing this module does not create files or load models."""

from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import getpass
import grp
import json
import os
import platform
import re
import shutil
import socket
import subprocess
from uuid import uuid4


PROJECT_ACCOUNT = "cis260330p"
SHARED_ROOT = Path("/ocean/projects/cis260330p/shared")


def repo_root():
    return Path(__file__).resolve().parents[2]


def _now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, payload):
    """Replace one JSON file atomically, leaving no temporary file on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path, payload):
    """Append one record. Each run has one writer; this is not a shared job log."""
    path = Path(path)
    line = json.dumps(payload, default=str, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(line)
        stream.flush()


def create_run(kind, config):
    """Create a unique personal results directory and snapshot configuration/provenance."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", kind):
        raise ValueError("Run kind must contain only letters, numbers, underscores or hyphens.")
    config = dict(config)
    username = getpass.getuser()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = repo_root() / "results" / f"{stamp}_{username}_{kind}_{uuid4().hex[:8]}"
    versions = {}
    for package in ("numpy", "Pillow", "torch", "diffusers", "transformers", "peft", "accelerate"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    git = {"commit": None, "dirty": None}
    try:
        command = ["git", "--no-optional-locks", "-C", str(repo_root())]
        git["commit"] = subprocess.check_output(command + ["rev-parse", "HEAD"], text=True,
                                                stderr=subprocess.DEVNULL, timeout=5).strip()
        git["dirty"] = bool(subprocess.check_output(command + ["status", "--porcelain"],
                                                    text=True, stderr=subprocess.DEVNULL, timeout=5))
    except (OSError, subprocess.SubprocessError):
        pass
    assets_path = Path(config.get("shared_root", SHARED_ROOT)) / "metadata" / "assets.json"
    assets = json.loads(assets_path.read_text()) if assets_path.is_file() else {}
    record = {
        "run_id": run_dir.name, "kind": kind, "status": "started", "started_utc": _now(),
        "username": username, "seed": config.get("seed"), "git": git,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"), "node": socket.gethostname(),
        "python": platform.python_version(), "packages": versions,
        "assets": {"model_revision": assets.get("model", {}).get("revision"),
                   "dataset_archive_sha256": assets.get("dataset", {}).get("archive_sha256")},
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "run.json", record)
    return run_dir


def finish_run(run_dir, status="completed", **fields):
    if status not in {"completed", "failed", "interrupted"}:
        raise ValueError("Final status must be completed, failed or interrupted.")
    path = Path(run_dir) / "run.json"
    record = json.loads(path.read_text())
    record.update(fields)
    record.update(status=status, finished_utc=_now())
    save_json(path, record)


def require_gpu():
    """Validate execution when called; never allocate a GPU or submit a job."""
    if Path("/ocean/projects/cis260330p").exists():
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Use an allocated compute-node Jupyter kernel for GPU cells.")
        if os.environ.get("SLURM_JOB_ACCOUNT") != PROJECT_ACCOUNT:
            raise RuntimeError(f"GPU cells require allocation -A {PROJECT_ACCOUNT}.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is visible in this notebook kernel.")
    return torch.device("cuda")


def record_image(run_dir, sample, prompt, target_color, method, seed, **fields):
    record = {"sample": sample, "prompt": prompt, "target_color": target_color,
              "method": method, "seed": seed, **fields}
    append_jsonl(Path(run_dir) / "images.jsonl", record)


def publish_run(run_dir, shared_root):
    """Explicitly copy completed artifacts to shared/runs/<user>; refuse replacement."""
    original = Path(run_dir)
    if original.is_symlink():
        raise ValueError("Publishing a symlink is not supported.")
    source = original.resolve(strict=True)
    if source.parent != (repo_root() / "results").resolve():
        raise ValueError("Publish only a run directly inside this repository's results directory.")
    record = json.loads((source / "run.json").read_text())
    if record.get("status") != "completed":
        raise ValueError("Complete the run before publishing it.")
    allowed = {".json", ".jsonl", ".csv", ".png", ".jpg", ".jpeg", ".webp", ".svg", ".safetensors"}
    sensitive_keys = re.compile(r"^(password|secret|api_key|private_key|access_token|hf_token|authorization|credentials)$", re.I)

    def check_payload(value):
        if isinstance(value, dict):
            if any(sensitive_keys.fullmatch(str(key)) for key in value):
                raise ValueError("Remove credentials from metadata before publishing.")
            for nested in value.values():
                check_payload(nested)
        elif isinstance(value, list):
            for nested in value:
                check_payload(nested)

    for item in source.rglob("*"):
        if item.is_symlink():
            raise ValueError("Run artifacts must not contain symbolic links.")
        if item.is_file() and item.suffix in {".json", ".jsonl"}:
            for payload in ([json.loads(item.read_text())] if item.suffix == ".json"
                            else [json.loads(line) for line in item.read_text().splitlines() if line.strip()]):
                check_payload(payload)
    group_id = grp.getgrnam(PROJECT_ACCOUNT).gr_gid
    root = Path(shared_root).resolve(strict=True)
    parent = root / "runs" / getpass.getuser()
    for directory in (root / "runs", parent):
        try:
            directory.mkdir(mode=0o2770)
        except FileExistsError:
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError(f"Shared publication path is not a directory: {directory}")
        else:
            os.chown(directory, -1, group_id)
            directory.chmod(0o2770)
    destination = parent / source.name
    destination.mkdir(mode=0o2770, exist_ok=False)
    try:
        os.chown(destination, -1, group_id)
        destination.chmod(0o2770)
        def ignored(directory, names):
            return [name for name in names if name.startswith(".") or
                    ((Path(directory) / name).is_file() and Path(name).suffix not in allowed)]
        shutil.copytree(source, destination, dirs_exist_ok=True, copy_function=shutil.copyfile, ignore=ignored)
        for item in [destination, *destination.rglob("*")]:
            os.chown(item, -1, group_id)
            item.chmod(0o2770 if item.is_dir() else 0o660)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return destination
