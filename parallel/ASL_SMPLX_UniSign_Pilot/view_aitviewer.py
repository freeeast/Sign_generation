"""View a generated SMPL-X clip or its approximate AITViewer skeleton."""
import argparse
from pathlib import Path

import numpy as np

BODY_EDGES = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),(8,10),
              (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]


def hand_edges(offset):
    return [(offset,offset+1+4*j) for j in range(5)] + [
        (offset+1+4*j+k,offset+2+4*j+k) for j in range(5) for k in range(3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clip',type=Path)
    parser.add_argument('--proxy',action='store_true',help='Use the matching file in proxy/ or specify its path.')
    parser.add_argument('--model-dir',type=Path,help='Parent of smplx/SMPLX_NEUTRAL.npz.')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    from aitviewer.viewer import Viewer

    if args.proxy:
        path = args.clip
        with np.load(path,allow_pickle=False) as data:
            if 'landmarks3d' not in data:
                path = Path(__file__).with_name('proxy')/path.name
        with np.load(path,allow_pickle=False) as data:
            points = data['landmarks3d']
            fps = float(data['fps'])
        from aitviewer.renderables.skeletons import Skeletons
        body = points[:,:17]
        lh = points[:,91:112]
        rh = points[:,112:133]
        nodes = [
            Skeletons(body,BODY_EDGES,radius=.009,color=(.22,.30,.41,1),name='Approximate body'),
            Skeletons(lh,hand_edges(0),radius=.004,color=(.02,.55,.59,1),name='Left hand'),
            Skeletons(rh,hand_edges(0),radius=.004,color=(.84,.30,.19,1),name='Right hand'),
        ]
    else:
        from aitviewer.configuration import CONFIG
        from aitviewer.models.smpl import SMPLLayer
        from aitviewer.renderables.smpl import SMPLSequence
        if not args.model_dir:
            parser.error('--model-dir is required for the SMPL-X mesh')
        asset=args.model_dir/'smplx'/'SMPLX_NEUTRAL.npz'
        if not asset.is_file():
            parser.error(f'Missing {asset}')
        CONFIG.update_conf({'smplx_models':str(args.model_dir.resolve())})
        layer=SMPLLayer(model_type='smplx',gender='neutral',use_pca=False,flat_hand_mean=True,
                        ext='npz',device=args.device)
        with np.load(args.clip,allow_pickle=False) as data:
            fps=float(data['fps'])
            nodes=[SMPLSequence(poses_body=data['body_pose'],smpl_layer=layer,poses_root=data['global_orient'],
                poses_left_hand=data['left_hand_pose'],poses_right_hand=data['right_hand_pose'],
                poses_jaw=data['jaw_pose'],expression=data['expression'],betas=data['betas'],
                trans=data['transl'],device=args.device,z_up=False,name=args.clip.stem)]
    viewer=Viewer()
    viewer.playback_fps=fps
    viewer.run_animations=True
    viewer.scene.camera.target=np.array([0,.30,0])
    viewer.scene.camera.position=np.array([0,.40,2.75])
    viewer.scene.add(*nodes)
    viewer.run()


if __name__=='__main__':
    main()
