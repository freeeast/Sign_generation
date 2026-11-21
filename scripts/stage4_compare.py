"""Source-held-out ranking against an external landmark detector, plus stability controls."""
import argparse
import csv
import hashlib
from itertools import combinations, product
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluate_feedback import write_csv

FEATURES = ("nll", "global_encoder", "frame_encoder", "hand_spatial", "hand_temporal")
CONDITIONS = ("baseline", "anchor_shift", "confidence07")


def answer(delta, threshold=1e-6):
    return int(np.sign(delta)) if abs(delta) > threshold else 0


def pairs_from(rows, cfg):
    cases = ("dose8", "alternative_from16", *[f"tangent_{s}" for s in cfg["tangent_seeds"][1:]])
    for clip, family in product(sorted({r["clip"] for r in rows}), cfg["families"]):
        group = {(r["condition"], r["case"]): r for r in rows if r["clip"] == clip and r["family"] == family}
        for ca, cb in combinations(cases, 2):
            for condition in CONDITIONS:
                a, b = group.get((condition, ca)), group.get((condition, cb))
                if not a or not b:
                    continue
                gap = lambda col: float(a[col]) - float(b[col])
                label_delta = gap("mp_B") if a["mp_B"] and b["mp_B"] else None
                valid = label_delta is not None and int(a["mp_B_frames"]) >= 8
                pair_id = hashlib.sha256(f"{clip}|{family}|{ca}|{cb}".encode()).hexdigest()[:12]
                yield dict(pair_id=pair_id, clip=clip, split=a["split"], family=family, condition=condition,
                           case_a=ca, case_b=cb, mp_frames=int(a["mp_B_frames"]),
                           label_kind="MediaPipe held-out 2D geometry",
                           mp_delta=label_delta if valid else None, label=answer(label_delta, cfg["detector_min_margin_px"]) if valid else 0,
                           geometry_A=gap("geometry_A"), geometry_B=gap("geometry_B"), **{f: gap(f) for f in FEATURES})


def accuracy(pairs, feature, scale, threshold):
    per_clip = {}
    predicted = []
    for row in pairs:
        geometry = row["geometry_A"]
        feedback = 0 if feature == "geometry" else np.clip(row[feature] / scale, -1., 1.)
        choice = answer(geometry / threshold + .5 * feedback)
        hit = 0.5 if choice == 0 else float(choice == row["label"])
        per_clip.setdefault(row["clip"], []).append(hit)
        predicted.append(dict(pair_id=row["pair_id"], clip=row["clip"], detector_label=row["label"],
                              method=feature, predicted=choice, hit=hit))
    return per_clip, predicted


def analyze(pairs, cfg, path):
    scales = {}
    for f in FEATURES:
        scales[f] = max(1e-6, float(np.median([abs(r[f]) for r in pairs
                      if r["split"] == "cal" and r["condition"] == "baseline"])))
    (path / "calibration_scales.json").write_text(json.dumps(scales, indent=2) + "\n")
    summary, all_predictions, clip_rows = [], [], []
    for threshold in (.5, cfg["geometry_tie_px"]):
        test = [r for r in pairs if r["split"] == "eval" and r["condition"] == "baseline"
                and abs(r["geometry_A"]) <= threshold and r["label"]]
        if not test:
            continue
        baseline, _ = accuracy(test, "geometry", 1., threshold)
        for method in ("geometry", *FEATURES):
            by_clip, predicted = accuracy(test, method, scales.get(method, 1.), threshold)
            all_predictions.extend(dict(threshold_px=threshold, **r) for r in predicted)
            source_means = [np.mean(v) for v in by_clip.values()]
            weighted = [x for values in by_clip.values() for x in values]
            diffs = [np.mean(by_clip[k]) - np.mean(baseline[k]) for k in by_clip]
            # Exact enumeration of the empirical source bootstrap. With three sources, descriptive only.
            boot = [np.mean(resample) for resample in product(diffs, repeat=len(diffs))]
            summary.append(dict(geometry_tie_px=threshold, method=method, n_pairs=len(test), n_sources=len(by_clip),
                                accuracy=np.mean(weighted), macro_accuracy=np.mean(source_means),
                                macro_delta_vs_geometry=np.mean(diffs), source_boot_lo=np.percentile(boot, 2.5),
                                source_boot_hi=np.percentile(boot, 97.5)))
            for clip, hits in by_clip.items():
                clip_rows.append(dict(geometry_tie_px=threshold, clip=clip, method=method,
                                      n=len(hits), correct=sum(hits), accuracy=np.mean(hits)))
    label_sensitivity = []
    for margin in (.5, 1.0):
        threshold = cfg["geometry_tie_px"]
        test = [r for r in pairs if r["split"] == "eval" and r["condition"] == "baseline"
                and abs(r["geometry_A"]) <= threshold and r["label"] and abs(r["mp_delta"]) > margin]
        for method in ("geometry", *FEATURES):
            if test:
                by_clip, _ = accuracy(test, method, scales.get(method, 1.), threshold)
                hits = [hit for values in by_clip.values() for hit in values]
                label_sensitivity.append(dict(mp_margin_px=margin, n_pairs=len(test), n_sources=len(by_clip),
                                              method=method, correct=sum(hits), accuracy=np.mean(hits)))
    if label_sensitivity:
        write_csv(path / "geometry_label_sensitivity.csv", label_sensitivity)
    for name, out in (("incremental_summary", summary), ("predictions", all_predictions), ("incremental_by_clip", clip_rows)):
        if out:
            write_csv(path / f"{name}.csv", out)


def controls(rows, cfg, path):
    result = []
    for clip, family, cond in product(sorted({r["clip"] for r in rows}), cfg["families"], CONDITIONS):
        bank = {r["case"]: r for r in rows if r["clip"] == clip and r["family"] == family and r["condition"] == cond}
        for comparison in ("corrupted16", "alternative_from16"):
            a, b = bank["dose8"], bank[comparison]
            result.append(dict(clip=clip, family=family, condition=cond, comparison=comparison,
                               **{f: float(a[f]) - float(b[f]) for f in (*FEATURES, "geometry_B", "mp_B")}))
    write_csv(path / "sensitivity.csv", result)

    stability = []
    for family, comparison, signal in product(cfg["families"], ("corrupted16", "alternative_from16"), FEATURES):
        cases = {(r["clip"], r["condition"]): answer(r[signal])
                 for r in result if r["family"] == family and r["comparison"] == comparison}
        for scope, clip_ids in (("all", cfg["calibration"]+cfg["evaluation"]), ("eval", cfg["evaluation"])):
            model = [cases[(clip, "baseline")] for clip in clip_ids]
            stability.append(dict(family=family, comparison=comparison, signal=signal, scope=scope,
                                n=len(clip_ids), prefers_restored=sum(x < 0 for x in model),
                                flip_anchor_shift=sum(cases[(clip, "anchor_shift")]*m<0 for clip,m in zip(clip_ids,model)),
                                flip_confidence07=sum(cases[(clip, "confidence07")]*m<0 for clip,m in zip(clip_ids,model))))
    write_csv(path / "sensitivity_summary.csv", stability)
    fig, axes = plt.subplots(1, len(cfg["families"]), figsize=(12, 4), layout="constrained")
    for ax, family in zip(axes, cfg["families"]):
        for signal in (*FEATURES, "geometry_A"):
            profiles = []
            for clip in cfg["evaluation"]:
                bank = {(r["family"], r["case"]): float(r[signal]) for r in rows if
                        r["condition"] == "baseline" and r["clip"] == clip and
                        (r["family"] in {family, "baseline"})}
                base = bank["baseline", "original"]
                if signal == "geometry_A" and family != "arm_trajectory":
                    with (Path("experiments/E3_5") / f"{clip}_local_probes.csv").open(newline="") as handle:
                        original = next(row for row in csv.DictReader(handle) if row["case"] == "original")
                    base = float(original["handshape_geometry_A_px"])
                values = np.array([0., *[bank[family, k] - base for k in ("dose3", "dose8", "corrupted16")]])
                profiles.append(values / max(np.max(np.abs(values)), 1e-8))
            if profiles:
                ax.plot([0, 3, 8, 16], np.median(profiles, axis=0), marker="o", label=signal)
        ax.axhline(0, c=".6", lw=.5)
        ax.set(title=family.replace("_", " "), xlabel="Perturbation (degrees)", ylabel="Scaled change in loss / distance")
    axes[-1].legend(fontsize=8, loc="best")
    fig.savefig(path / "perturbation_response.png", dpi=160)


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("path", type=Path)
    path = arg.parse_args().path
    cfg = json.loads(Path("configs/stage4.json").read_text())
    with (path / "feedback.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    pairs = list(pairs_from(rows, cfg))
    write_csv(path / "candidate_pairs.csv", pairs)
    analyze(pairs, cfg, path)
    controls(rows, cfg, path)


if __name__ == "__main__":
    main()
