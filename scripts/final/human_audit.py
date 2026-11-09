"""Build an offline, blinded annotation page and analyze returned human CSVs.

No model inference, training, or image export is performed by this module.
The export script supplies already prepared region crops and a private key.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import uuid
import zipfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "results/final-color/human-audit"
HEADER = ["pack_id", "annotator_id", "item_id", "human_color", "clothing_pattern"]
BASE_COLORS = {"black", "brown", "red", "blue", "green", "purple", "pink", "white", "gray"}
NON_CLEAR = {"mixed", "uncertain"}
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def load_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_color(value, region):
    """Only normalize documented palette aliases; never guess a human color."""
    if value is None:
        return None
    value = str(value).strip().lower()
    value = {"grey": "gray", "blonde": "blond"}.get(value, value)
    if region == "hair" and value == "yellow":
        value = "blond"
    elif region == "cloth" and value == "blond":
        value = "yellow"
    return value


def palette(region):
    return BASE_COLORS | {"blond" if region == "hair" else "yellow"} | NON_CLEAR


def validate_key(path):
    key = load_json(path)
    if not isinstance(key, dict) or not IDENTIFIER.fullmatch(str(key.get("pack_id", ""))):
        raise ValueError("Private key requires a safe pack_id.")
    items = key.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Private key must have a nonempty items list.")
    index = {}
    for source in items:
        item = dict(source)
        item_id = str(item.get("item_id", ""))
        if not IDENTIFIER.fullmatch(item_id) or item_id in index:
            raise ValueError(f"Invalid or duplicate item_id: {item_id!r}")
        if item.get("region") not in {"hair", "cloth"}:
            raise ValueError(f"Invalid region for {item_id}.")
        if item.get("kind") not in {"source", "edited"}:
            raise ValueError(f"Invalid kind for {item_id}; use source or edited.")
        for field in ("lab_color", "clip_color", "cnn_color", "target_color"):
            if item.get(field) is not None:
                item[field] = normalize_color(item[field], item["region"])
                if item[field] not in palette(item["region"]) - NON_CLEAR:
                    raise ValueError(f"Invalid {field} for {item_id}: {item[field]!r}")
        if item["kind"] == "edited" and not item.get("target_color"):
            raise ValueError(f"Edited item {item_id} requires target_color.")
        index[item_id] = item
    return key, index


def validate_local_image(pack_dir, value):
    if not isinstance(value, str) or not value:
        raise ValueError("An image path must be a relative path.")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or ":" in value or "\\" in value:
        raise ValueError(f"Unsafe image path: {value}")
    path = (pack_dir / relative).resolve()
    if not path.is_relative_to(pack_dir.resolve()) or not path.is_file():
        raise ValueError(f"Pack image is missing or outside pack: {value}")
    return relative.as_posix()


def build_pack(key_path, pack_dir):
    """Write only blinded HTML/instructions; never copy private labels into it."""
    key, index = validate_key(key_path)
    pack_dir = Path(pack_dir)
    if not pack_dir.is_dir():
        raise ValueError(f"Export the images to this directory first: {pack_dir}")
    if Path(key_path).resolve().is_relative_to(pack_dir.resolve()):
        raise ValueError("Private answer key must be OUTSIDE the public annotation pack.")
    public_manifest = load_json(pack_dir / "manifest.json")
    if public_manifest.get("pack_id") != key["pack_id"]:
        raise ValueError("Public manifest and private key have different pack_id values.")
    public_items = []
    public_ids = set()
    for item in public_manifest.get("items", []):
        item_id = item.get("item_id")
        if item_id not in index or item_id in public_ids or item.get("region") != index[item_id]["region"]:
            raise ValueError(f"Public/private manifest mismatch for {item_id!r}.")
        public_ids.add(item_id)
        public = {"item_id": item_id, "region": item["region"],
                  "image": validate_local_image(pack_dir, item["image"])}
        if item.get("context_image"):
            public["context_image"] = validate_local_image(pack_dir, item["context_image"])
        public_items.append(public)
    if public_ids != set(index):
        raise ValueError("Public manifest does not cover all private-key items.")
    # Exporter has shuffled and anonymized these IDs; keep that exact order.
    public = {"schema_version": 1, "pack_id": key["pack_id"], "items": public_items}
    # Keep the downloadable manifest strictly blinded too.
    write_json(pack_dir / "manifest.json", public)
    encoded = json.dumps(public, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    token = "/*__AUDIT_DATA__*/null"
    if template.count(token) != 1:
        raise RuntimeError("Annotation HTML template has an invalid data placeholder.")
    (pack_dir / "index.html").write_text(template.replace(token, encoded), encoding="utf-8")
    (pack_dir / "INSTRUCTIONS.txt").write_text(
        "颜色标注：给完全不懂项目的标注者\n\n"
        "1. 下载整个压缩包，先解压；不要在压缩包内直接打开文件。\n"
        "2. 双击 index.html，用 Chrome / Edge / Firefox / Safari 打开，不需要联网。\n"
        "   不要重命名或单独移动图片，保持解压后的 images 文件夹和网页放在一起。\n"
        "3. 填组织者分配的代号（例如 A01），不要填真实姓名；点击开始。\n"
        "4. 每张只看指定的头发或衣服区域，选实际看见的颜色。灰色背景不计入。\n"
        "   没有一种明显主色选 mixed（多色）；看不清或拿不准选 uncertain（无法判断）。\n"
        "   衣服另选纯色、花纹或无法判断。光照、褶皱、头发反光不单独算花纹。\n"
        "   这是判断实际颜色，不是猜测别人想改成什么颜色；不需要评价是否自然。\n"
        "5. 点击下一张，直到进度显示全部完成。不要与其他标注人商量答案。\n"
        "6. 点击下载标注 CSV；把下载的 annotations_...csv 原文件发回组织者。\n"
        "   不要只发截图，不需要手工编辑 CSV，也不要把网页当作已提交。\n"
        "7. 网页会尝试在本浏览器保存进度；换电脑/浏览器或清理浏览器数据可能丢失。\n"
        "   中途要退出，先下载 CSV 备份，下次填相同代号后可导入 CSV 继续。\n\n"
        f"本任务包：{key['pack_id']}；共 {len(index)} 张。\n"
        "建议保持同一屏幕和正常色彩设置，关闭偏黄色的夜间模式。\n\n"
        "English: Unzip the whole folder, open index.html, enter your assigned anonymous ID,\n"
        "judge the selected region's actual color, answer clothing pattern when applicable,\n"
        "and download and return the CSV. No login/network/install is needed.\n",
        encoding="utf-8")
    archive = pack_dir.parent / "annotation-pack.zip"
    files = {"index.html", "manifest.json", "INSTRUCTIONS.txt"}
    files.update(item[field] for item in public_items for field in ("image", "context_image") if item.get(field))
    temporary = archive.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for relative in sorted(files):
            zipped.write(pack_dir / relative, relative)
    temporary.replace(archive)
    write_json(archive.with_suffix(".zip.sha256.json"), {
        "pack_id": key["pack_id"], "zip_file": archive.name,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "public_manifest_sha256": hashlib.sha256((pack_dir / "manifest.json").read_bytes()).hexdigest(),
        "n_items": len(public_items), "private_key_included": False,
    })
    (pack_dir.parent / "responses").mkdir(exist_ok=True)
    print(f"Blinded annotation page ready: {pack_dir / 'index.html'}", flush=True)
    print(f"Download this archive to your Mac and unzip: {archive}", flush=True)
    print(f"Return downloaded annotations_*.csv to: {pack_dir.parent / 'responses'}", flush=True)
    return public


def response_files(paths):
    files = []
    for raw in paths:
        path = Path(raw)
        files.extend(sorted(path.glob("*.csv")) if path.is_dir() else [path])
    unique = sorted(set(p.resolve() for p in files))
    if not unique:
        raise ValueError("No returned CSVs found. Download the annotator's CSV first.")
    if any(not p.is_file() for p in unique):
        raise ValueError("A response path does not exist.")
    return unique


def read_responses(paths, pack_id, items):
    """Combine identical duplicate snapshots; reject conflicting answers explicitly."""
    answers = defaultdict(dict)
    sources = defaultdict(set)
    for path in response_files(paths):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != HEADER:
                raise ValueError(f"{path.name}: required exact CSV columns: {','.join(HEADER)}")
            seen = set()
            for line, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"{path.name}:{line}: invalid CSV column count.")
                if row["pack_id"] != pack_id:
                    raise ValueError(f"{path.name}:{line}: wrong pack_id.")
                annotator = row["annotator_id"].strip()
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", annotator):
                    raise ValueError(f"{path.name}:{line}: invalid annotator_id.")
                item_id = row["item_id"]
                if item_id not in items:
                    raise ValueError(f"{path.name}:{line}: unknown item_id {item_id!r}.")
                identity = (annotator, item_id)
                if identity in seen:
                    raise ValueError(f"{path.name}:{line}: duplicate annotator/item row.")
                seen.add(identity)
                item = items[item_id]
                color = normalize_color(row["human_color"], item["region"])
                pattern = row["clothing_pattern"].strip()
                sources[annotator].add(str(path))
                answers[annotator]  # Retain an entirely blank submission for completeness checks.
                if not color:
                    if pattern and not (item["region"] == "cloth" and pattern in {"plain", "patterned", "uncertain"}):
                        raise ValueError(f"{path.name}:{line}: invalid partial pattern.")
                    continue
                if color not in palette(item["region"]):
                    raise ValueError(f"{path.name}:{line}: invalid human_color {color!r}.")
                allowed = {"not_applicable"} if item["region"] == "hair" else {"plain", "patterned", "uncertain", ""}
                if pattern not in allowed:
                    raise ValueError(f"{path.name}:{line}: invalid clothing_pattern {pattern!r}.")
                value = {"human_color": color, "clothing_pattern": pattern}
                previous = answers[annotator].get(item_id)
                if previous is not None and previous != value:
                    # Updating a partial clothing response to a complete one is safe.
                    if previous["human_color"] == color and (not previous["clothing_pattern"] or not pattern):
                        value = previous if previous["clothing_pattern"] else value
                    else:
                        raise ValueError(f"Conflicting answers for {annotator}/{item_id}. Keep only the intended final CSV.")
                answers[annotator][item_id] = value
    return answers, sources


def ratio(numerator, denominator):
    return {"correct": numerator, "n": denominator,
            "fraction": numerator / denominator if denominator else None}


def summarize_rows(rows):
    clear = [row for row in rows if row["human_color"] not in NON_CLEAR]
    metrics = {"n_annotated": len(rows), "n_clear": len(clear),
               "n_mixed": sum(r["human_color"] == "mixed" for r in rows),
               "n_uncertain": sum(r["human_color"] == "uncertain" for r in rows),
               "proxy_agreement_with_clear_human": {}}
    for proxy in ("lab_color", "clip_color", "cnn_color"):
        eligible = [r for r in clear if r.get(proxy)]
        metrics["proxy_agreement_with_clear_human"][proxy] = ratio(
            sum(r[proxy] == r["human_color"] for r in eligible), len(eligible))
    edited = [r for r in rows if r["kind"] == "edited"]
    if edited:
        definite = [r for r in edited if r["human_color"] not in NON_CLEAR]
        non_uncertain = [r for r in edited if r["human_color"] != "uncertain"]
        correct = sum(r["human_color"] == r["target_color"] for r in edited)
        metrics["human_target_match_clear_only"] = ratio(correct, len(definite))
        metrics["human_target_match_including_mixed_excluding_uncertain"] = ratio(correct, len(non_uncertain))
        metrics["human_target_match_all_answered_conservative"] = ratio(correct, len(edited))
    return metrics


def aggregate(rows):
    groups = {"all": rows}
    for field in ("kind", "region", "method", "target_color", "clothing_pattern"):
        for value in sorted({str(r.get(field, "")) for r in rows if r.get(field)}):
            groups[f"{field}={value}"] = [r for r in rows if str(r.get(field, "")) == value]
    for kind in ("source", "edited"):
        for region in ("hair", "cloth"):
            subset = [r for r in rows if r["kind"] == kind and r["region"] == region]
            groups[f"{kind}/{region}"] = subset
            if region == "cloth":
                for pattern in ("plain", "patterned", "uncertain"):
                    groups[f"{kind}/{region}/{pattern}"] = [r for r in subset if r["clothing_pattern"] == pattern]
            if kind == "edited":
                for method in sorted({r.get("method", "unknown") for r in subset}):
                    group = [r for r in subset if r.get("method", "unknown") == method]
                    groups[f"{kind}/{region}/method={method}"] = group
                    for target in sorted({r["target_color"] for r in group}):
                        groups[f"{kind}/{region}/method={method}/target={target}"] = [r for r in group if r["target_color"] == target]
    return {name: summarize_rows(group) for name, group in groups.items()}


def load_cnn_predictions(path, items):
    """Optional exported CSV: item_id,predicted_color; other fields are allowed."""
    if path is None:
        return
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"item_id", "predicted_color"}.issubset(reader.fieldnames or []):
            raise ValueError("CNN CSV must include item_id,predicted_color columns.")
        seen = set()
        for row in reader:
            item_id = row["item_id"]
            if item_id not in items or item_id in seen:
                raise ValueError(f"CNN predictions contain unknown or duplicate item_id {item_id!r}.")
            seen.add(item_id)
            color = normalize_color(row["predicted_color"], items[item_id]["region"])
            if color not in palette(items[item_id]["region"]) - NON_CLEAR:
                raise ValueError(f"Invalid CNN color for {item_id}.")
            items[item_id]["cnn_color"] = color


def join_cnn_root(root, items):
    """Join classifier predictions without exposing them to the annotator."""
    root = Path(root)
    source = {}
    edited = {}
    provenance = []
    source_path = root / "source_predictions.jsonl"
    paths = ([source_path] if source_path.is_file() else []) + sorted(root.glob("evaluations/*/cnn_metrics.jsonl"))
    for path in paths:
        provenance.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                region = row.get("region")
                if region not in {"hair", "cloth"}:
                    raise ValueError(f"{path}:{line_number}: invalid CNN region.")
                image_id = int(row["image_id"])
                color = normalize_color(row["cnn_color"], region)
                if color not in palette(region) - NON_CLEAR:
                    raise ValueError(f"{path}:{line_number}: invalid CNN color.")
                if path == source_path:
                    identity = (image_id, region)
                    container = source
                else:
                    identity = (path.parent.name, str(row["method"]), image_id, region,
                                normalize_color(row["target_color"], region))
                    container = edited
                if identity in container and container[identity] != color:
                    raise ValueError(f"Conflicting CNN predictions for {identity!r}.")
                container[identity] = color
    matched = 0
    for item in items.values():
        image_id = int(item["source_image_id"])
        if item["kind"] == "source":
            color = source.get((image_id, item["region"]))
        else:
            run = Path(str(item.get("run_id", ""))).name
            color = edited.get((run, str(item.get("method", "")), image_id,
                                item["region"], item["target_color"]))
        if color is not None:
            item["cnn_color"] = color
            matched += 1
    return {"root": str(root.resolve()), "files": provenance, "matched_items": matched,
            "total_items": len(items)}


def analyze(key_path, response_paths, output=None, cnn_predictions=None,
            cnn_root=ROOT / "results/final-color/cnn", allow_incomplete=False,
            allow_missing_cnn=False):
    key, items = validate_key(key_path)
    cnn_provenance = join_cnn_root(cnn_root, items)
    load_cnn_predictions(cnn_predictions, items)
    answers, sources = read_responses(response_paths, key["pack_id"], items)
    if not answers:
        raise ValueError("No answered color labels found; no report was written.")
    missing_cnn = sorted(item_id for item_id, item in items.items() if not item.get("cnn_color"))
    if missing_cnn and not allow_missing_cnn:
        raise ValueError(f"CNN predictions are missing for {len(missing_cnn)}/{len(items)} audit items. "
                         "Complete CNN evaluation first, or use --allow-missing-cnn for a clearly partial audit report.")
    report = {"schema_version": 1, "pack_id": key["pack_id"],
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "key_sha256": hashlib.sha256(Path(key_path).read_bytes()).hexdigest(),
              "cnn_predictions": cnn_provenance,
              "missing_cnn_item_ids": missing_cnn,
              "notes": ["Human labels are held-out audit observations, never training labels.",
                        "Agreement is against each annotator, not verified objective ground truth.",
                        "Mixed and uncertain labels are excluded from clear-color proxy agreement; coverage is reported.",
                        "No majority vote is invented for two annotators; pairwise differences are retained.",
                        "A convenience audit sample does not estimate whole-dataset performance without sampling assumptions."],
              "annotators": {}, "pairwise": []}
    expanded_rows = []
    for annotator in sorted(answers):
        rows = [{**items[item_id], **value, "annotator_id": annotator}
                for item_id, value in answers[annotator].items()]
        missing = sorted(set(items) - set(answers[annotator]))
        incomplete_pattern = sorted(r["item_id"] for r in rows if r["region"] == "cloth" and not r["clothing_pattern"])
        report["annotators"][annotator] = {"complete": not missing and not incomplete_pattern,
                                         "missing_item_ids": missing,
                                         "incomplete_pattern_item_ids": incomplete_pattern,
                                         "response_files": sorted(sources[annotator]),
                                         "groups": aggregate(rows)}
        expanded_rows.extend(rows)
    incomplete = [name for name, result in report["annotators"].items() if not result["complete"]]
    if incomplete and not allow_incomplete:
        details = "; ".join(f"{name}: {len(report['annotators'][name]['missing_item_ids'])} missing colors, "
                            f"{len(report['annotators'][name]['incomplete_pattern_item_ids'])} missing clothing patterns"
                            for name in incomplete)
        raise ValueError(f"Incomplete submission(s): {details}. Finish and export CSV again. "
                         "Use --allow-incomplete only for an explicitly partial progress report.")
    disagreements = []
    for left, right in itertools.combinations(sorted(answers), 2):
        shared = sorted(set(answers[left]) & set(answers[right]))
        clear = [i for i in shared if answers[left][i]["human_color"] not in NON_CLEAR and answers[right][i]["human_color"] not in NON_CLEAR]
        disagree = [i for i in shared if answers[left][i]["human_color"] != answers[right][i]["human_color"]]
        both_cloth = [i for i in shared if items[i]["region"] == "cloth" and answers[left][i]["clothing_pattern"] and answers[right][i]["clothing_pattern"]]
        report["pairwise"].append({"annotator_a": left, "annotator_b": right,
                                  "color_agreement_all_answered": ratio(len(shared) - len(disagree), len(shared)),
                                  "color_agreement_both_clear": ratio(sum(answers[left][i]["human_color"] == answers[right][i]["human_color"] for i in clear), len(clear)),
                                  "clothing_pattern_agreement": ratio(sum(answers[left][i]["clothing_pattern"] == answers[right][i]["clothing_pattern"] for i in both_cloth), len(both_cloth)),
                                  "disagreement_item_ids": disagree})
        for item_id in disagree:
            disagreements.append({"item_id": item_id, "annotator_a": left, "color_a": answers[left][item_id]["human_color"],
                                  "annotator_b": right, "color_b": answers[right][item_id]["human_color"]})
    output = Path(output) if output else DEFAULT_ROOT / "analysis" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", report)
    fields = ["annotator_id", "item_id", "kind", "region", "method", "target_color", "human_color", "clothing_pattern", "lab_color", "clip_color", "cnn_color"]
    with (output / "joined_labels.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(expanded_rows)
    with (output / "disagreements.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item_id", "annotator_a", "color_a", "annotator_b", "color_b"])
        writer.writeheader()
        writer.writerows(disagreements)
    lines = [f"# Human color audit: {key['pack_id']}", "",
             "Per-person results; no assumed consensus. Percentages below use only explicit labels.", ""]
    for annotator, entry in report["annotators"].items():
        lines += [f"## Annotator {annotator}", "", f"Status: {'COMPLETE' if entry['complete'] else 'INCOMPLETE — do not describe as a completed audit'}.",
                  f"Missing color answers: {len(entry['missing_item_ids'])}; missing clothing-pattern answers: {len(entry['incomplete_pattern_item_ids'])}.", "",
                  "| Group | Answered | Clear | Mixed | Uncertain | Lab agreement | CLIP agreement | CNN agreement | Human target match (clear only) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for group, stats in entry["groups"].items():
            if group == "all" or "/" not in group:
                continue
            def display(value):
                return f"{value['correct']}/{value['n']} ({100 * value['fraction']:.1f}%)" if value and value["n"] else "—"
            proxy = stats["proxy_agreement_with_clear_human"]
            lines.append(f"| {group} | {stats['n_annotated']} | {stats['n_clear']} | {stats['n_mixed']} | {stats['n_uncertain']} | {display(proxy['lab_color'])} | {display(proxy['clip_color'])} | {display(proxy['cnn_color'])} | {display(stats.get('human_target_match_clear_only'))} |")
        lines.append("")
    lines += ["## Interpretation", "", "- This is an audit on the selected examples, not a new training dataset.",
              "- Mixed and uncertain cases are reported, not silently mapped to a single color.",
              "- Clear-only target match can be optimistic; summary.json also reports mixed-inclusive and conservative all-answered denominators.",
              "- Per-annotator subgroup sample sizes must accompany reported percentages.",
              "- Review disagreements.csv if two people differ; report agreement before any adjudication.",
              "- CNN trained on automatic labels is a diagnostic evaluator; this audit alone does not make it objective ground truth."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Audit analysis saved: {output}", flush=True)
    for annotator, entry in report["annotators"].items():
        print(f"{annotator}: {'complete' if entry['complete'] else 'INCOMPLETE'}; "
              f"{len(answers[annotator])}/{len(items)} color answers", flush=True)
    if not missing_cnn and all(entry["complete"] for entry in report["annotators"].values()):
        print("FINAL_AUDIT_COMPLETE: all submitted annotators are complete and every audit item has a CNN prediction.", flush=True)
    else:
        print(f"PARTIAL_AUDIT: missing CNN predictions={len(missing_cnn)}; incomplete annotators={len(incomplete)}.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-pack", aliases=["build"], help="Build offline HTML and ZIP from already exported images.")
    build.add_argument("--key", type=Path, default=DEFAULT_ROOT / "key.json")
    build.add_argument("--pack", type=Path, default=DEFAULT_ROOT / "pack")
    audit = commands.add_parser("analyze", help="Compare returned human CSVs with the private answer key.")
    audit.add_argument("--key", type=Path, default=DEFAULT_ROOT / "key.json")
    audit.add_argument("--responses", type=Path, nargs="+", default=[DEFAULT_ROOT / "responses"])
    audit.add_argument("--cnn-predictions", type=Path)
    audit.add_argument("--cnn-root", type=Path, default=ROOT / "results/final-color/cnn")
    audit.add_argument("--allow-incomplete", action="store_true")
    audit.add_argument("--allow-missing-cnn", action="store_true")
    audit.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command in {"build", "build-pack"}:
            build_pack(args.key, args.pack)
        else:
            analyze(args.key, args.responses, args.output, args.cnn_predictions,
                    args.cnn_root, args.allow_incomplete, args.allow_missing_cnn)
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(2, f"Human audit error: {error}\n")


if __name__ == "__main__":
    main()
