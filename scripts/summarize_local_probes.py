"""Summarize nominal restorations, local dose response, and frame-split ranking utility."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.evaluate_feedback import write_csv
from src.probe_3d import ROTATION_GROUPS


def classify(delta, tolerance):
    return np.sign(delta).item() if abs(delta) > tolerance else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    files = sorted(root.glob("*_local_probes.csv"))
    if not files:
        raise ValueError("No completed local probes found")
    clips = {}
    for file in files:
        with file.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
            clips[rows[0]["clip"]] = {(r["family"], r["case"]): r for r in rows}
    contrasts, dose_summary = [], []
    for family in (*ROTATION_GROUPS, "temporal_warp"):
        temporal = family == "temporal_warp"
        fine, mid, large = (1, 3, 7) if temporal else (3, 8, 16)
        alternative = "alternative_reverse_shift" if temporal else "alternative_from16"
        suffix = "handshape" if family in {"finger_articulation", "wrist_orientation"} else "tip"
        for clip, track in clips.items():
            baseline = track["baseline", "original"]
            small = track[family, f"dose{fine}"]
            negative = track[family, f"negative{fine}"]
            restored = track[family, f"restored_from{large}_to{mid}"]
            corrupted = track[family, f"corrupted{large}"]
            d_nll = np.array([float(r["nll"]) - float(baseline["nll"]) for r in (small, restored, corrupted)])
            def geo(row, fold):
                return float(row[f"{suffix}_geometry_{fold}_px"])
            dose_summary.append(dict(clip=clip, family=family, small_magnitude=fine,
                delta_nll_small=d_nll[0], delta_nll_mid=d_nll[1], delta_nll_large=d_nll[2],
                strictly_increasing_nll=bool(np.all(np.diff(np.r_[0, d_nll]) > 1e-4)),
                monotonic_from_small=bool(np.all(np.diff(d_nll) > 1e-4)),
                both_small_directions_raise_nll=bool(d_nll[0] > 1e-4 and float(negative["nll"]) - float(baseline["nll"]) > 1e-4),
                central_difference_nll=(float(small["nll"]) - float(negative["nll"]))/(2*fine),
                small_heldout_geometry_px=geo(small,"B") - geo(baseline,"B")))
            for candidate in (corrupted, track[family, alternative]):
                delta = float(restored["nll"]) - float(candidate["nll"])
                cos_delta = float(restored["cosine_to_real"]) - float(candidate["cosine_to_real"])
                g_a, g_b = [geo(restored, f) - geo(candidate, f) for f in ("A","B")]
                contrasts.append(dict(clip=clip, family=family, control=candidate["case"], geometry=suffix,
                    restored_minus_control_nll=delta, restored_minus_control_cosine=cos_delta,
                    restored_minus_control_geometry_A_px=g_a, restored_minus_control_geometry_B_px=g_b))
    write_csv(root / "correction_contrasts.csv", contrasts)
    write_csv(root / "dose_response_summary.csv", dose_summary)
    families = []
    for family in (*ROTATION_GROUPS, "temporal_warp"):
        true_restore = [r for r in contrasts if r["family"] == family and r["control"].startswith("corrupted")]
        alternative = [r for r in contrasts if r["family"] == family and r["control"].startswith("alternative")]
        doses = [r for r in dose_summary if r["family"] == family]
        families.append(dict(family=family, n=len(true_restore),
            nll_prefers_restored=sum(r["restored_minus_control_nll"] < -1e-4 for r in true_restore),
            cosine_prefers_restored=sum(r["restored_minus_control_cosine"] > 1e-4 for r in true_restore),
            geometry_B_prefers_restored=sum(r["restored_minus_control_geometry_B_px"] < -.1 for r in true_restore),
            median_restored_minus_corrupt_nll=float(np.median([r["restored_minus_control_nll"] for r in true_restore])),
            nll_prefers_restored_to_alternative=sum(r["restored_minus_control_nll"] < -1e-4 for r in alternative),
            dose_strictly_increases_nll=sum(r["strictly_increasing_nll"] for r in doses),
            small_nll_degradation_count=sum(r["delta_nll_small"] > 1e-4 for r in doses),
            median_delta_nll_small=float(np.median([r["delta_nll_small"] for r in doses])),
            both_small_directions_raise_nll=sum(r["both_small_directions_raise_nll"] for r in doses)))
    write_csv(root / "family_summary.csv", families)

    tie_results = []
    for r in contrasts:
        g_a, g_b = (r[f"restored_minus_control_geometry_{k}_px"] for k in ("A","B"))
        outcome = classify(g_b, .1)
        nll_choice = classify(r["restored_minus_control_nll"], 1e-4)
        geometry_choice = classify(g_a, 1e-9)
        if abs(g_a) <= .5 and outcome and geometry_choice and nll_choice:
            tie_results.append(dict(r, geometry_only_correct=bool(geometry_choice == outcome),
                                    nll_tiebreak_correct=bool(nll_choice == outcome),
                                    cosine_tiebreak_correct=bool(
                                        classify(-r["restored_minus_control_cosine"], 1e-4) == outcome)))
    if tie_results:
        write_csv(root / "geometry_near_ties.csv", tie_results)
    total = dict(clips=len(clips), scenarios=sum(len(r) for r in clips.values()),
                 contrasts=len(contrasts), near_tie_count=len(tie_results),
                 near_tie_geometry_correct=sum(r["geometry_only_correct"] for r in tie_results),
                 near_tie_nll_correct=sum(r["nll_tiebreak_correct"] for r in tie_results),
                 near_tie_cosine_correct=sum(r["cosine_tiebreak_correct"] for r in tie_results))
    (root / "probe_totals.json").write_text(json.dumps(total, indent=2) + "\n")
    print(json.dumps(total, indent=2))


if __name__ == "__main__":
    main()
