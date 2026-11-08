import argparse
import csv
import json
import statistics as st
from pathlib import Path

SHARED = "/ocean/projects/cis260330p/shared/runs/wsun8"
MIDTERM = "20261019T163222971419Z_wsun8_evaluate_2745b5d9"
SEEDS = {m: [m, m + "-s36", m + "-s37"] for m in ("E1", "Main", "P25", "Light", "LightP25", "NoBlue", "E1v", "Mainv")}

# per eval run: clip, summary, kid and arcface by (method, region, target)
runs = {}
# each method's main-protocol eval run; b0 and b1 come from the midterm run
where = {}
PALETTE = []


def is_palette(name):
    targets = runs[name]["cfg"].get("evaluation", {}).get("targets", {})
    return sum(len(v) for v in targets.values()) > 6


def load(run_dirs):
    eval_dirs = sorted(str(p) for d in run_dirs for p in Path(d).glob("*_evaluate_*"))
    metric_dirs = sorted(str(p) for d in run_dirs for pattern in ("*_arcface_*", "*_fid_*") for p in Path(d).glob(pattern))
    for d in eval_dirs:
        name = Path(d).name
        cfg = json.load(open(Path(d) / "config.json"))
        r = runs[name] = {"clip": {}, "summ": {}, "kid": {}, "arc": {}, "cfg": cfg}
        for row in json.load(open(Path(d) / "clip_summary.json"))["table"]:
            r["clip"][(row["method"], row["region"], row["target"])] = row
        for row in csv.DictReader(open(Path(d) / "summary.csv")):
            r["summ"][(row["method"], row["region"], row["target"])] = row
    for d in metric_dirs:
        cfg = json.load(open(Path(d) / "config.json"))
        name = Path(cfg["eval_run"]).name
        if name not in runs:
            continue
        if "_fid_" in Path(d).name:
            for row in csv.DictReader(open(Path(d) / "fid_summary.csv")):
                runs[name]["kid"][(row["method"], row["region"], row["target"])] = 1e3 * float(row["kid_region"])
        else:
            for row in csv.DictReader(open(Path(d) / "arcface_summary.csv")):
                runs[name]["arc"][(row["method"], row["region"], row["target"])] = float(row["identity_cos"])
    for name in runs:
        if is_palette(name):
            continue
        for (m, _, _) in runs[name]["summ"]:
            if m in ("B0", "B1"):
                where[m] = MIDTERM
            else:
                where.setdefault(m, name)
    PALETTE.extend(n for n in runs if is_palette(n))


def get(metric, name, region, target="all", run=None):
    r = runs[run or where[name]]
    key = (name, region, target)
    if metric == "clip":
        return 100 * r["clip"][key]["clip_color_accuracy"]
    if metric == "cs":
        return r["clip"][key]["clip_score"]
    if metric == "lab":
        return 100 * float(r["summ"][key]["color_accuracy"])
    if metric == "de":
        return float(r["summ"][key]["delta_e"])
    if metric == "lpips":
        return float(r["summ"][key]["lpips_outside"])
    if metric == "mae":
        return float(r["summ"][key]["mae_outside"])
    if metric == "kid":
        return r["kid"][key]
    return r["arc"][key]


def have(method):
    return all(n in where for n in SEEDS.get(method, [method]))


def cell(metric, method, region, target="all", fmt="{:.1f}", std=True, run=None):
    try:
        xs = [get(metric, n, region, target, run) for n in SEEDS.get(method, [method])]
    except KeyError:
        return "--"
    if len(xs) == 1 or not std:
        return "$" + fmt.format(st.mean(xs)) + "$"
    return "$" + fmt.format(st.mean(xs)) + "\\pm" + fmt.format(st.stdev(xs)) + "$"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=[SHARED])
    load(parser.parse_args().runs)

    print("% eval runs:", {n[-8:]: ("palette" if is_palette(n) else sorted({k[0] for k in runs[n]["summ"]})) for n in runs})
    print("% missing metrics:", [n[-8:] for n in runs if not runs[n]["kid"] or not runs[n]["arc"]])

    print("% b0 and b1 in every eval run that has them (clip acc, lab acc, lpips, kid, arcface; hair then cloth)")
    for m in ("B0", "B1"):
        for name in runs:
            if is_palette(name) or (m, "hair", "all") not in runs[name]["summ"]:
                continue
            vals = []
            for region in ("hair", "cloth"):
                for k in ("clip", "lab", "lpips", "kid", "arc"):
                    try:
                        vals.append(round(get(k, m, region, run=name), 6))
                    except KeyError:
                        vals.append(None)
            print("%", m, name[-8:], vals)

    # same adapters evaluated on l40s (midterm runs) and on v100 (rerun)
    v100_b1 = [n for n in runs if n != MIDTERM and not is_palette(n) and ("B1", "hair", "all") in runs[n]["summ"]
               and runs[n]["cfg"].get("adapters", {}).get("Mainv")]
    if v100_b1:
        print("% hardware check: l40s vs v100 (clip, lab, de, cs, arc, lpips, kid)")
        for region in ("hair", "cloth"):
            for m, mv in (("B1", "B1"), ("E1", "E1v"), ("Main", "Mainv")):
                if not have(mv) and mv != "B1":
                    continue
                row = []
                for k, f in (("clip", "{:.1f}"), ("lab", "{:.1f}"), ("de", "{:.1f}"), ("cs", "{:.2f}"),
                             ("arc", "{:.3f}"), ("lpips", "{:.3f}"), ("kid", "{:.1f}")):
                    try:
                        a = st.mean(get(k, n, region) for n in SEEDS.get(m, [m]))
                        b = (get(k, "B1", region, run=v100_b1[0]) if mv == "B1"
                             else st.mean(get(k, n, region) for n in SEEDS[mv]))
                        row.append(f"{k} {f.format(a)}->{f.format(b)}")
                    except KeyError:
                        row.append(f"{k} --")
                print("%", region, m, " | ".join(row))

    print("% main table")
    for region in ("hair", "cloth"):
        for m in ("B0", "B1", "E1", "Main", "P25", "Light", "LightP25"):
            if not have(m):
                continue
            row = [cell("clip", m, region), cell("lab", m, region), cell("de", m, region),
                   cell("cs", m, region, fmt="{:.2f}", std=False), cell("arc", m, region, fmt="{:.3f}", std=False),
                   cell("lpips", m, region, fmt="{:.3f}", std=False), cell("kid", m, region)]
            print(region, m, " & ".join(row), "\\\\")

    print("% per target")
    for region, targets in (("hair", ("blond", "red", "blue")), ("cloth", ("red", "blue", "white"))):
        for t in targets:
            for m in ("B0", "B1", "E1", "Main", "P25", "Light", "LightP25"):
                if not have(m):
                    continue
                row = [cell("clip", m, region, t, std=False), cell("lab", m, region, t, std=False),
                       cell("lpips", m, region, t, fmt="{:.3f}", std=False), cell("kid", m, region, t, std=False)]
                print(region, t, m, " & ".join(row), "\\\\")

    print("% seed 35 sensitivity")
    for m in ("E1", "P25", "Main", "P75", "rank4", "rank64", "data10", "data50", "Light", "LightP25"):
        if m not in where:
            continue
        # single seed-35 runs, so e1 and main are not averaged over seeds
        row = []
        for k, reg, f in (("clip", "hair", "{:.1f}"), ("lab", "hair", "{:.1f}"), ("kid", "hair", "{:.1f}"),
                          ("arc", "hair", "{:.3f}"), ("clip", "cloth", "{:.1f}"), ("kid", "cloth", "{:.1f}"),
                          ("lpips", "cloth", "{:.3f}")):
            try:
                row.append("$" + f.format(get(k, m, reg)) + "$")
            except KeyError:
                row.append("--")
        print(m, " & ".join(row), "\\\\")

    print("% withheld colors: blue hair, blue cloth, red hair, red cloth")
    for m in ("B1", "E1", "Main", "NoBlue", "NoRed"):
        if not have(m):
            continue
        row = [cell("clip", m, reg, t) for reg, t in (("hair", "blue"), ("cloth", "blue"), ("hair", "red"), ("cloth", "red"))]
        print(m, " & ".join(row), "\\\\")

    print("% withheld colors, other targets and costs: blond hair, white cloth, hair kid, cloth kid, hair lpips, cloth lpips")
    for m in ("Main", "NoBlue", "NoRed"):
        if not have(m):
            continue
        row = [cell("clip", m, "hair", "blond"), cell("clip", m, "cloth", "white"), cell("kid", m, "hair"),
               cell("kid", m, "cloth"), cell("lpips", m, "hair", fmt="{:.3f}", std=False),
               cell("lpips", m, "cloth", fmt="{:.3f}", std=False)]
        print(m, " & ".join(row), "\\\\")

    print("% mae outside, seed means")
    for region in ("hair", "cloth"):
        print("%", region, {m: round(st.mean(get("mae", n, region) for n in SEEDS.get(m, [m])), 4)
                            for m in ("B0", "B1", "E1", "Main", "P25", "Light", "LightP25") if have(m)})

    for name in PALETTE:
        r = runs[name]
        methods = sorted({k[0] for k in r["summ"]}, key=lambda m: ("B0", "B1", "E1", "Main", "Light", "LightP25").index(m)
                         if m in ("B0", "B1", "E1", "Main", "Light", "LightP25") else 99)
        print("% palette", name[-8:], "clip acc / lab acc per target")
        for region in ("hair", "cloth"):
            targets = r["cfg"]["evaluation"]["targets"][region]
            for t in targets:
                row = []
                for m in methods:
                    try:
                        row.append(f"{get('clip', m, region, t, name):.0f}/{get('lab', m, region, t, name):.0f}")
                    except KeyError:
                        row.append("--")
                print("%", region, t, dict(zip(methods, row)))
        # pooled over the edits of each group, so colors with more edits weigh more
        synthetic = {"red", "blue", "green", "purple", "pink", "brown"}
        print("% palette groups: clip acc, lab acc, lpips (synthetic palette then other colors)")
        for region in ("hair", "cloth"):
            targets = r["cfg"]["evaluation"]["targets"][region]
            for gname, group in (("synthetic", [t for t in targets if t in synthetic]),
                                 ("other", [t for t in targets if t not in synthetic])):
                row = []
                for m in methods:
                    try:
                        n = [int(r["summ"][(m, region, t)]["n"]) for t in group]
                        clip = sum(k * get("clip", m, region, t, name) for k, t in zip(n, group)) / sum(n)
                        lab = sum(k * get("lab", m, region, t, name) for k, t in zip(n, group)) / sum(n)
                        lp = sum(k * get("lpips", m, region, t, name) for k, t in zip(n, group)) / sum(n)
                        row.append(f"${clip:.1f}$ & ${lab:.1f}$ & ${lp:.3f}$")
                    except KeyError:
                        row.append("--")
                print(region, gname, "+".join(group), sum(int(r["summ"][(methods[0], region, t)]["n"]) for t in group),
                      " | ".join(f"{m}: {c}" for m, c in zip(methods, row)))


if __name__ == "__main__":
    main()
