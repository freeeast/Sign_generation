"""Forward-kinematics diagnostic using an approximate rig, then project COCO-WholeBody."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from generate import COCO_FINGERS, FINGERS, SIDE


def forward(pose):
    body = pose['body_pose'].reshape(-1,21,3)
    n = len(body)
    p = np.zeros((n,133,3),dtype=np.float32)
    mask = np.zeros(133,dtype=bool)
    head = Rotation.from_rotvec(body[:,11])*Rotation.from_rotvec(body[:,14])
    head_center = np.array([0,.62,.010])
    root = Rotation.from_rotvec(pose['global_orient'])

    def put(index, value):
        p[:,index] = value
        mask[index] = True

    def face(index, local):
        put(index,head_center + head.apply(local))

    face(0, [0,.005,.104])
    for side, body_indices, brain_indices, start in [
        ('left',(15,17,19),(5,7,9),(1,3,11,13,15,91)),
        ('right',(16,18,20),(6,8,10),(2,4,12,14,16,112))]:
        s = SIDE[side]
        eye, ear, hip, knee, ankle, hand_offset = start
        face(eye, [s*.035,.029,.086])
        face(ear, [s*.078,.015,.025])
        put(hip, [s*.09,0,0])
        put(knee, [s*.095,-.42,.02])
        put(ankle, [s*.095,-.82,.01])
        shoulder = np.broadcast_to([s*.18,.46,0], (n,3))
        r_sh, r_el, r_wr = [Rotation.from_rotvec(body[:,i]) for i in body_indices]
        r_el = r_sh*r_el
        r_wr = r_el*r_wr
        elbow = shoulder + r_sh.apply([s*.285,-.018,-.006])
        wrist = elbow + r_el.apply([s*.255,-.012,0])
        for idx,value in zip(brain_indices, (shoulder,elbow,wrist)):
            put(idx,value)
        put(hand_offset,wrist)
        finger_angles = pose[f'{side}_hand_pose'].reshape(n,5,3,3)
        for j,name in enumerate(COCO_FINGERS):
            slot = FINGERS.index(name)
            spread = (.034,.021,0.,-.019,-.037)[j]
            base_length = (.035,.073,.079,.074,.063)[j]
            lens = ((.034,.026,.022),(.041,.025,.019),(.046,.029,.021),
                    (.043,.027,.019),(.033,.022,.017))[j]
            joint = wrist+r_wr.apply([s*base_length, -.014 if j==0 else 0, spread])
            rot = r_wr
            put(hand_offset+1+4*j, joint)
            for k,length in enumerate(lens):
                rot = rot*Rotation.from_rotvec(finger_angles[:,slot,k])
                joint = joint+rot.apply([s*length, 0, (.18*length if j==0 else 0)])
                put(hand_offset+2+4*j+k,joint)

    for j,t in enumerate(np.linspace(np.pi,2*np.pi,17)):
        face(23+j, [.081*np.cos(t),.105*np.sin(t)+.017,.043-.018*abs(np.cos(t))])
    face(53, [0,-.021,.096])
    mouth = pose['jaw_pose'][:,0]*.04
    for j,t in enumerate(np.linspace(np.pi,3*np.pi,8,endpoint=False)):
        points = np.repeat(np.array([[.031*np.cos(t),-.048+.009*np.sin(t),.086]]),n,axis=0)
        if np.sin(t)<0:
            points[:,1] -= mouth
        face(83+j, points)

    p[:,mask] = root.apply(p[:,mask].reshape(-1,3)).reshape(n,-1,3) if n == 1 else np.einsum(
        'fij,fnj->fni',root.as_matrix(),p[:,mask])
    p[:,mask] += pose['transl'][:,None,:]
    return p, mask


def convert(path, out_dir, static_dir, confidence):
    with np.load(path,allow_pickle=False) as motion:
        landmarks, mask = forward(motion)
        fps=float(motion['fps'])
    xy=landmarks[:,:,:2]*[1,-1]
    xy[:,mask]=xy[:,mask]*440+[320,355]
    score=np.broadcast_to(mask,(len(xy),133)).astype('float32')*confidence
    np.savez_compressed(out_dir/path.name,keypoints=xy.astype('float32'),scores=score,
                        landmarks3d=landmarks,mapped=mask,fps=np.float32(fps))
    if static_dir is not None:
        np.savez_compressed(static_dir/path.name,keypoints=np.repeat(xy[:1],len(xy),axis=0).astype('float32'),
                            scores=score,fps=np.float32(fps))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('poses',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--static-dir',type=Path)
    parser.add_argument('--confidence',type=float,default=.90)
    args=parser.parse_args()
    if not .3 < args.confidence <= 1:
        parser.error('--confidence must be > .3 and <= 1')
    args.output.mkdir(exist_ok=True,parents=True)
    if args.static_dir is not None:
        args.static_dir.mkdir(exist_ok=True,parents=True)
    for path in sorted(args.poses.glob('S*.npz')):
        convert(path,args.output,args.static_dir,args.confidence)


if __name__=='__main__':
    main()
