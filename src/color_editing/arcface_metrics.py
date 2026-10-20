"""ArcFace identity similarity between each edited image and its original (T1).

Reads a finished evaluation run (cases, saved images for every method) and
writes a new run under results/ with one cosine similarity per edit. The face
is detected once on the original photo; that same alignment is applied to
every method's output, so recolored hair cannot move or lose the crop.
The shared evaluation run is only read, never modified.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .data import load_sample
from .evaluate import case_name
from .runs import append_jsonl, create_run, finish_run, repo_root, save_json

MODEL_PACK = "buffalo_l"
MODEL_ROOT = repo_root() / "results" / "_insightface_cache"
# CelebA-HQ faces fill the 512 frame; the detector's usual 640 input upscales them
# past its anchor range and scores drop below threshold. Native size first, smaller second.
DET_SIZES = ((512, 512), (320, 320))
BATCH = 32


def thread_count():
    """Cap ONNX Runtime threads; its default of one per core thrashes a shared login node."""
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    return max(1, min(int(os.environ.get("ARCFACE_THREADS", 8)), cores))


def load_models(providers=None):
    import onnxruntime
    from insightface.app import FaceAnalysis

    if not (MODEL_ROOT / "models" / MODEL_PACK).is_dir():
        raise FileNotFoundError(f"Download {MODEL_PACK} into {MODEL_ROOT} on a login node first.")
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = thread_count()
    options.inter_op_num_threads = 1
    app = FaceAnalysis(name=MODEL_PACK, root=str(MODEL_ROOT), providers=providers, sess_options=options,
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0 if providers and "CUDAExecutionProvider" in providers else -1, det_size=DET_SIZES[0])
    return app.det_model, app.models["recognition"]


def pick_providers():
    import onnxruntime
    available = onnxruntime.get_available_providers()
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]


def to_bgr(image):
    return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()


def primary_face(det, bgr):
    """Largest detected face; CelebA-HQ crops are centred on one subject."""
    for size in DET_SIZES:
        bboxes, kpss = det.detect(bgr, input_size=size)
        if bboxes.shape[0]:
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            i = int(np.argmax(areas))
            return {"bbox": [float(v) for v in bboxes[i, :4]], "det_score": float(bboxes[i, 4]),
                    "kps": kpss[i], "det_size": list(size)}
    return None


def embed(rec, bgr_images, kps):
    from insightface.utils import face_align

    aligned = [face_align.norm_crop(img, landmark=kps, image_size=rec.input_size[0]) for img in bgr_images]
    feats = np.asarray(rec.get_feat(aligned), dtype=np.float32)
    return feats / np.linalg.norm(feats, axis=1, keepdims=True)


def score_run(eval_run, limit=None, methods=None):
    eval_run = Path(eval_run).resolve(strict=True)
    config = json.loads((eval_run / "config.json").read_text())
    cases = json.loads((eval_run / "cases.json").read_text())
    if limit:
        cases = cases[:limit]
    found = sorted(p.name for p in (eval_run / "images").iterdir() if p.is_dir())
    methods = [m for m in found if not methods or m in methods]
    providers = pick_providers()
    run = create_run("arcface", {"eval_run": str(eval_run), "shared_root": config["shared_root"],
                                 "mask_dilation": config["mask_dilation"], "model_pack": MODEL_PACK,
                                 "det_sizes": [list(s) for s in DET_SIZES], "providers": providers,
                                 "threads": thread_count(), "methods": methods, "limit": limit})
    print(f"{len(cases)} cases, methods {methods}, providers {providers}, threads {thread_count()}", flush=True)
    try:
        rows, faces, misses = _score(eval_run, config, cases, methods, run, providers)
        table = summarize(rows, run)
        finish_run(run, cases=len(cases), methods=methods, faces_found=len(faces), faces_missed=misses)
    except BaseException as error:
        finish_run(run, status="failed", error=f"{type(error).__name__}: {error}")
        raise
    print(f"finished {run}", flush=True)
    return run, table


def _score(eval_run, config, cases, methods, run, providers):
    det, rec = load_models(providers)
    data_root = Path(config["shared_root"]) / "data/CelebAMask-HQ"

    # one detection per original photo and region, reused for every method
    samples, faces, misses = {}, {}, []
    started = time.perf_counter()
    for c in cases:
        key = (c["image_id"], c["region"])
        if key in samples:
            continue
        sample = load_sample(data_root, c["image_id"], c["region"], config["mask_dilation"])
        bgr = to_bgr(sample["image"])
        face = primary_face(det, bgr)
        samples[key] = {"bgr": bgr, "kps": None, "feat": None}
        if face is None:
            misses.append({"image_id": c["image_id"], "region": c["region"]})
            continue
        samples[key].update(kps=face["kps"], feat=embed(rec, [bgr], face["kps"])[0])
        faces[key] = {"image_id": c["image_id"], "region": c["region"], **face, "kps": face["kps"].tolist()}
    save_json(run / "faces.json", {"found": list(faces.values()), "missed": misses})
    print(f"faces: {len(faces)} found, {len(misses)} missed, "
          f"{time.perf_counter() - started:.1f}s", flush=True)

    rows = []
    for method in methods:
        started = time.perf_counter()
        for start in range(0, len(cases), BATCH):
            batch = cases[start:start + BATCH]
            # group by original so one alignment call covers all its edits
            by_key = {}
            for c in batch:
                by_key.setdefault((c["image_id"], c["region"]), []).append(c)
            for key, group in by_key.items():
                s = samples[key]
                if s["kps"] is None:
                    for c in group:
                        rows.append({**c, "method": method, "face_found": False, "identity_cos": None})
                        append_jsonl(run / "arcface_metrics.jsonl", rows[-1])
                    continue
                outs = [to_bgr(Image.open(eval_run / "images" / method / f"{case_name(c)}.png")) for c in group]
                feats = embed(rec, outs, s["kps"])
                for c, f in zip(group, feats):
                    rows.append({**c, "method": method, "face_found": True,
                                 "identity_cos": float(np.dot(f, s["feat"]))})
                    append_jsonl(run / "arcface_metrics.jsonl", rows[-1])
        print(f"{method}: {len(cases)} edits in {time.perf_counter() - started:.1f}s", flush=True)
    return rows, faces, len(misses)


def summarize(rows, run):
    groups = {}
    for r in rows:
        for key in ((r["method"], r["region"], "all"), (r["method"], r["region"], r["target_color"])):
            groups.setdefault(key, []).append(r)
    table = []
    for (method, region, target), items in sorted(groups.items()):
        scored = [r["identity_cos"] for r in items if r["identity_cos"] is not None]
        table.append({"method": method, "region": region, "target": target, "n": len(items),
                      "n_scored": len(scored),
                      "identity_cos": float(np.mean(scored)) if scored else None,
                      "identity_cos_std": float(np.std(scored)) if scored else None,
                      "identity_cos_p10": float(np.percentile(scored, 10)) if scored else None})
    save_json(run / "arcface_summary.json", table)
    with (run / "arcface_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for row in table:
        if row["target"] == "all":
            cos = "   n/a" if row["identity_cos"] is None else f"{row['identity_cos']:.4f}"
            p10 = "   n/a" if row["identity_cos_p10"] is None else f"{row['identity_cos_p10']:.4f}"
            print(f"{row['method']:>16} {row['region']:>5} n={row['n']:4d} scored={row['n_scored']:4d} "
                  f"identity_cos={cos} p10={p10}", flush=True)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-run", required=True, help="finished evaluation run with cases.json and images/")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N cases (smoke test)")
    parser.add_argument("--methods", nargs="*", default=None, help="subset of image folders to score")
    args = parser.parse_args()
    score_run(args.eval_run, limit=args.limit, methods=args.methods)
