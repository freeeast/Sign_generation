"""Audit wrist correspondence and detect high-confidence out-of-frame predictions."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.stage4_review_media import label, pose_on
from src.evaluate_feedback import write_csv
from src.pose import LEFT, RIGHT
from src.virtual_pose import TIPS


def audit_rgb(root, threshold):
    rows = []
    for directory in sorted((root/"rgb").iterdir()):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*_detector.npz")):
            with np.load(path) as data:
                xy, scores = data["keypoints"],data["scores"]
            inside = ((xy>=0)&(xy<=1)).all(axis=-1)
            for side,ids in (("left",LEFT),("right",RIGHT)):
                s = scores[:,ids]>.3
                i = inside[:,ids]
                rows.append(dict(clip=directory.name,method=path.stem.removesuffix("_detector"),hand=side,
                    over_threshold_and_outside_fraction=float((s&~i).mean()),
                    frames_qualified_with_in_frame_joint_fraction=float(((s&i).sum(axis=1)>=threshold).mean()),
                    median_confidence_inside=float(np.median(scores[:,ids][i])) if i.any() else None))
        with np.load(directory/"real_detector.npz") as raw, np.load(directory/"avatar_detector.npz") as rendered:
            real, virtual = raw["keypoints"], rendered["keypoints"]
            # Both sides on screen and large real-to-rendered fingertip mismatch.
            comparable = ((real[:,TIPS]>=0)&(real[:,TIPS]<=1)&(virtual[:,TIPS]>=0)&(virtual[:,TIPS]<=1)).all(axis=-1)
            offsets = np.linalg.norm((real[:,TIPS]-virtual[:,TIPS])*[1280,720],axis=-1)
            frame_id = int(np.argmax(np.where(comparable,offsets,0.).mean(axis=1)))
        image_pair(directory, frame_id)
    write_csv(root/"rgb/image_bounds.csv",rows)


def image_pair(directory, frame):
    tiles = []
    for name in ("real","avatar"):
        capture=cv2.VideoCapture(str(directory/f"{name}.mp4"))
        capture.set(cv2.CAP_PROP_POS_FRAMES,frame)
        ok,img=capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"Expected video {name} frame {frame}")
        with np.load(directory/f"{name}_detector.npz") as data:
            xy, s=data["keypoints"][frame],data["scores"][frame]
        pose_on(img,np.where(s[:,None]>.3,xy,np.nan)*[img.shape[1],img.shape[0]],(30,220,250))
        img=cv2.resize(img,(800,450))
        cv2.rectangle(img,(0,0),(800,31),(28,28,28),-1)
        label(img,f"{name}   |   visible hand discrepancy, frame {frame}",(10,20))
        tiles.append(img)
    cv2.imwrite(str(directory/"visible_hand_discrepancy.png"),np.concatenate(tiles,axis=1))
    (directory/"visible_hand_discrepancy.json").write_text(json.dumps({"frame_zero_based":frame})+"\n")


def audit_wrist(root):
    rows=[]
    for paired in sorted((root/"data/paired").glob("*.npz")):
        with np.load(paired) as pair, np.load(root/"data/real_pose"/paired.name) as raw:
            confidence=raw["scores"]
            y=np.where(confidence[...,None]>.3,raw["keypoints"],np.nan)
            wh=np.array([raw["width"],raw["height"]])
            body=y[:,[9,10]]
            hands=y[:,[91,112]]
            smpl=pair["original2d"][:,[91,112]]
            mp=pair["mp"][:,[91,112]]
        pairs={"fit_to_rtmw_body":(smpl,body),"fit_to_rtmw_hand":(smpl,hands),"rtmw_body_to_hand":(body,hands),
               "fit_to_mp":(smpl,mp),"rtmw_hand_to_mp":(hands,mp),"fit_to_swapped_rtmw_hand":(smpl,hands[:,::-1])}
        for name,(x,z) in pairs.items():
            for s, side in enumerate(("left","right")):
                residual=(x[:,s]-z[:,s])*wh
                residual=residual[np.isfinite(residual).all(axis=1)]
                rows.append(dict(clip=paired.stem,metric=name,hand=side,n=len(residual),
                    median_px=np.median(np.linalg.norm(residual,axis=-1)),
                    median_signed_x_px=np.median(residual[:,0]),median_signed_y_px=np.median(residual[:,1])))
    write_csv(root/"data/wrist_correspondence.csv",rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    cfg=json.loads(Path("configs/stage6.json").read_text())["rgb"]
    audit_wrist(args.output)
    audit_rgb(args.output,cfg["min_hand_landmarks_over_threshold"])


if __name__=="__main__":
    main()
