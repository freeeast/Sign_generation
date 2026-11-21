"""Project the official SMPL-X forward pass to frozen Uni-Sign input."""
import argparse
from pathlib import Path

import numpy as np

from src.smplx_to_pose import forward


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path,help='Directory with canonical per-clip NPZ rotations.')
    p.add_argument('output',type=Path)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--gender',choices=('neutral','male','female'),default='neutral')
    p.add_argument('--hand-mean',choices=('flat','model'),default='flat')
    p.add_argument('--confidence',type=float,default=.90)
    p.add_argument('--mask-face',action='store_true')
    p.add_argument('--right',type=float,nargs=3,default=(1.,0.,0.))
    p.add_argument('--down',type=float,nargs=3,default=(0.,-1.,0.))
    args=p.parse_args()
    right,down=np.array(args.right),np.array(args.down)
    if not np.isclose(np.linalg.norm(right),1.) or not np.isclose(np.linalg.norm(down),1.) or not np.isclose(right@down,0.):
        p.error('right and down must be orthogonal unit vectors')
    if args.batch_size < 1 or not .3 < args.confidence <= 1:
        p.error('Use positive batch size and confidence in (.3,1]')
    args.output.mkdir(exist_ok=True,parents=True)
    for path in sorted(args.input.glob('S*.npz')):
        points,mapped=forward(path,args.model_dir,args.gender,args.hand_mean,args.batch_size,
                              args.device,args.mask_face)
        xy=np.float32(np.stack((points@right,points@down),axis=-1)*440+[320,355])
        scores=np.float32(np.broadcast_to(mapped,xy.shape[:2])*args.confidence)
        np.savez_compressed(args.output/path.name,keypoints=xy,scores=scores)


if __name__=='__main__':
    main()
