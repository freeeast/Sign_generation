"""Audit candidate accounting and local joint angles against the original fitted reference."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.stage5_closed_loop import ARMS, SIGNALS
from src.evaluate_feedback import write_csv
from src.stage5_motion import MotionSpace, rotation
from src.virtual_pose import load_fit


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def original_relative_angles(fit, space):
    rows = {}
    for family, group in space.group.items():
        relative = []
        for field, indices in group.items():
            observed = fit[field].reshape(space.n,-1,3) + space.means.get(field, 0)
            delta = (rotation(space.original[field][:,indices]).inv()*rotation(observed[:,indices])).magnitude()
            relative.extend(np.rad2deg(delta).tolist())
        rows[family+"_median_deg"] = float(np.median(relative))
        rows[family+"_p95_deg"] = float(np.percentile(relative,95))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("root","motions","asset"):
        parser.add_argument("--"+option,required=True,type=Path)
    parser.add_argument("--config",type=Path,default=Path("configs/stage5.json"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    scales = json.loads((args.root/"calibration_scales.json").read_text())
    assert set(scales)==set(SIGNALS) and all(np.isfinite(x) and x>0 for x in scales.values())
    rotation_errors = []
    summary = []
    steps_count = len(cfg["sweeps_deg"])*len(cfg["corruption_coeff_deg"])
    holdout = read(args.root/"comparison.csv")
    assert len(holdout) == 3*(len(ARMS)+2)
    for clip in (cfg["calibration_clip"],*cfg["evaluation_clips"]):
        directory = args.root/clip
        source = load_fit(args.motions/f"{clip}.npz")
        space = MotionSpace(source,args.asset,cfg)
        budget = set()
        accepted_count = 0
        n_violations = 0
        for method in ("I_perturbed", *ARMS):
            name = "I_perturbed_parameters.npz" if method=="I_perturbed" else f"{method}_final_parameters.npz"
            ending = load_fit(directory/name)
            assert all(np.array_equal(source[k], ending[k]) for k in source.keys()-space.edited.keys())
            rotation_errors.append(dict(clip=clip,method=method,**original_relative_angles(ending,space)))
            if method=="I_perturbed":
                continue
            coeff = np.load(directory/f"{method}_coefficients.npy",allow_pickle=False)
            assert coeff.shape==(steps_count+1,len(cfg["corruption_coeff_deg"])) and not np.any(coeff[0])
            assert np.max(np.abs(coeff)) <= cfg["max_coeff_deg"]
            candidate, _ = space.candidate(coeff[-1])
            for field in space.edited:
                np.testing.assert_allclose(ending[field],candidate[field],atol=1e-12)
            trajectory = read(directory/f"{method}_trajectory.csv")
            proposals = read(directory/f"{method}_proposals.csv")
            n_violations += sum(int(x["constraint_violations"])>0 for x in proposals)
            assert len(trajectory)==steps_count+1 and len(proposals)==2*steps_count
            previous = None
            accepted = 0
            for i,r in enumerate(trajectory):
                value = float(r["objective"])
                assert int(r["constraint_violations"])==0
                if i:
                    accepted_here = not np.array_equal(coeff[i],coeff[i-1])
                    assert accepted_here == (r["accepted"]=="True")
                    assert value < previous if accepted_here else abs(value-previous)<1e-10
                    accepted += accepted_here
                previous = value
            budget.add(len(proposals))
            accepted_count += accepted
        assert budget == {36}
        summary.append(dict(clip=clip, logical_proposals=36*len(ARMS),
                            rejected_by_limits=n_violations, accepted_updates=accepted_count))
    write_csv(args.root/"audit_counts.csv",summary)
    write_csv(args.root/"original_fit_rotation_audit.csv",rotation_errors)
    print(summary)


if __name__=="__main__":
    main()
