"""Use archived same-motion renders to isolate camera and appearance sensitivity."""
import argparse
from pathlib import Path

import numpy as np

from src.evaluate_feedback import write_csv
from src.stage7_feedback import (HandGraphAdapter, HandGraphs, confidence_diagnostics, cosine_error,
                                 expert_input, feature_scores, hand_quality, valid_pose)


def run(a):
    graph=HandGraphs(HandGraphAdapter(a.repo,a.checkpoint,"cpu",a.threads))
    d=Path("experiments/Stage6/rgb/3xni-I6N3EY_3-1-rgb_front")
    def load(key):
        with np.load(d/f"{key}_detector.npz") as p:
            return p["keypoints"][::2],p["scores"][::2]
    xy,c=load("real")
    valid=valid_pose(xy,c,.3)
    strong=valid_pose(xy,c,.45)
    qualified=hand_quality(valid)
    strong_quality=hand_quality(strong)
    ref=graph.extract(xy,c)
    ref_mask=graph.extract(*expert_input(xy,c,valid))
    ref_strong=graph.extract(*expert_input(xy,c,strong))
    rows=[]
    for mode in ("real","avatar","clay","camera_shift"):
        v,score=load(mode)
        p=valid_pose(v,score,.3)
        s=valid_pose(v,score,.45)
        raw=graph.extract(v,score)
        values=cosine_error(raw[1].reshape(len(v),2,-1),ref[1].reshape(len(v),2,-1))
        masked=graph.extract(*expert_input(v,score,valid&p))
        conf=graph.extract(*expert_input(v,score,strong&s))
        m=feature_scores(masked,ref_mask,qualified,hand_quality(p),np.ones_like(qualified,float),1.)
        conf_score=feature_scores(conf,ref_strong,strong_quality,hand_quality(s),np.ones_like(qualified,float),1.)
        rows.append(dict(condition=mode,legacy_right_temporal=values[:,1].mean(),bilateral_temporal=values.mean(),
                         fixed_support_spatial=m[0],fixed_support_temporal=m[1],fixed_support_multiscale=m[2],
                         strong_confidence_temporal=conf_score[1],fixed_support=m[3],reference_hand_frames=int(qualified.sum()),
                         **confidence_diagnostics(v,score,.3)))
    write_csv(a.output/"same_motion_nuisance.csv",rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("repo","checkpoint","output"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--threads",type=int,default=2)
    run(p.parse_args())


if __name__=="__main__":
    main()
