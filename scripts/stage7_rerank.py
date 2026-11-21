"""Geometry-anchored abstention and oracle diagnostics on the exact same proposal bank."""
import argparse
import csv
from pathlib import Path

from scripts.stage7_optimize import GEOM_KEYS, read
from src.evaluate_feedback import write_csv


def run(root):
    cfg=read(Path("configs/stage7.json"))
    scales=read(root/"calibration/scales.json")
    margin=.5*scales["adapted"]
    rows=[]
    for d in sorted((root/"direct").glob("*")):
        path=d/"robust_geometry_proposals.csv"
        if not path.exists():
            continue
        baseline=read(d/"robust_geometry.json")
        proposals=list(csv.DictReader(path.open()))
        available=[r for r in proposals if r["rejection"]=="" and r["mp_held_combined"]]
        for slack_scale in (0.,1.):
            feasible=[r for r in available if all(
                float(r[k])<=baseline[k]+slack_scale*cfg["geometry_slack_px"][k]+1e-8 for k in GEOM_KEYS)
                and float(r["support_retained"])>=baseline["support_retained"]-.05]
            positive=[r for r in feasible if float(r["adapted"])<baseline["adapted"]-margin]
            chosen=min(positive,key=lambda r:float(r["adapted"])) if positive else None
            oracle=min((float(r["mp_held_combined"]) for r in feasible),default=baseline["mp_held_combined"])
            rows.append(dict(clip=d.name,slack_multiplier=slack_scale,n_proposals=len(proposals),
                n_kinematically_valid=len(available),n_feasible=len(feasible),n_expert_margin=len(positive),
                abstained=chosen is None,expert_delta=float(chosen["adapted"])-baseline["adapted"] if chosen else 0.,
                heldout_delta=float(chosen["mp_held_combined"])-baseline["mp_held_combined"] if chosen else 0.,
                oracle_heldout_delta=min(0.,oracle-baseline["mp_held_combined"]),
                selected_coefficients=chosen["coefficients"] if chosen else None))
    write_csv(root/"safe_rerank.csv",rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    run(p.parse_args().root)


if __name__=="__main__":
    main()
