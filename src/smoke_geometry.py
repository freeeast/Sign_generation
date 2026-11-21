"""Exercise the official FK and our mapping using a synthetic kinematic fixture."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from .project_keypoints import orthographic, plot_frames
from .smplx_to_pose import forward, map_landmarks, COCO_TO_SMPLX


def synthetic_asset(output):
    from smplx.vertex_ids import vertex_ids

    # Artificial geometry, shaped solely to test tensor dimensions and projection.
    vertices = 10475
    template = np.zeros((vertices, 3), dtype=np.float32)
    template[16:22] = [[.15, .30, 0], [-.15, .30, 0], [.25, .18, 0],
                       [-.25, .18, 0], [.18, .17, 0], [-.18, .17, 0]]
    for side, start in (("left", 25), ("right", 40)):
        direction = 1 if side == "left" else -1
        for j, name in enumerate(("index", "middle", "pinky", "ring", "thumb")):
            x = direction * (.18 + j * .009)
            template[start + 3*j:start + 3*j + 3] = [[x, .21+.02*k, 0.] for k in range(3)]
            template[vertex_ids["smplx"][side[0] + name]] = [x, .27, 0.]
    for key, point in {"nose": (0, .45, 0), "leye": (.025, .47, 0),
                       "reye": (-.025, .47, 0), "lear": (.06, .44, 0),
                       "rear": (-.06, .44, 0)}.items():
        template[vertex_ids["smplx"][key]] = point
    angles = np.linspace(0, 2*np.pi, 68, endpoint=False)
    template[200:268, :2] = np.stack((.045*np.cos(angles), .44+.048*np.sin(angles)), axis=1)
    faces = np.stack([np.arange(200, 268), np.full(68, 270), np.full(68, 271)], axis=1)
    weights = np.zeros((vertices, 55), dtype=np.float32)
    weights[:, 0] = 1.
    kintree = np.stack((np.zeros(55, dtype=int), np.arange(55)))
    coords = np.repeat([[1., 0., 0.]], 68, axis=0)
    np.savez_compressed(
        output, v_template=template, shapedirs=np.zeros((vertices, 3, 20), dtype=np.float32),
        posedirs=np.zeros((vertices, 3, 486), dtype=np.float32),
        J_regressor=np.eye(55, vertices, dtype=np.float32), weights=weights, kintree_table=kintree,
        f=faces, lmk_faces_idx=np.arange(51), lmk_bary_coords=coords[:51],
        dynamic_lmk_faces_idx=np.tile(np.arange(51, 68), (79, 1)),
        dynamic_lmk_bary_coords=np.tile(coords[51:], (79, 1, 1)),
        hands_componentsl=np.eye(45), hands_componentsr=np.eye(45),
        hands_meanl=np.zeros(45), hands_meanr=np.zeros(45),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from smplx.joint_names import JOINT_NAMES

    t = 5
    assert len(COCO_TO_SMPLX) == 133
    codes = np.tile(np.arange(len(JOINT_NAMES))[None, :, None], (t, 1, 3))
    indexed, mask = map_landmarks(codes, JOINT_NAMES, mask_face=True)
    assert indexed[0, 91, 0] == JOINT_NAMES.index("left_wrist")
    assert indexed[0, 112, 0] == JOINT_NAMES.index("right_wrist")
    assert not mask[23:91].any() and not indexed[:, 23:91].any()
    with tempfile.TemporaryDirectory() as folder:
        model_path = Path(folder) / "smplx" / "SMPLX_NEUTRAL.npz"
        model_path.parent.mkdir()
        synthetic_asset(model_path)
        motion = Path(folder) / "motion.npz"
        translation = np.linspace(0., 0.04, t)
        np.savez(motion, global_orient=np.zeros((t, 3)), body_pose=np.zeros((t, 63)),
                 left_hand_pose=np.zeros((t, 45)), right_hand_pose=np.zeros((t, 45)),
                 transl=np.stack((translation, np.zeros(t), np.zeros(t)), axis=-1))
        points, mapped = forward(motion, Path(folder), "neutral", "flat", 3, "cpu", False)
    assert points.shape == (t, 133, 3) and mapped.all()
    np.testing.assert_allclose(points[-1] - points[0], np.tile([.04, 0., 0.], (133, 1)), atol=1e-6)
    image = orthographic(points, mapped, (1, 0, 0), (0, -1, 0), 1, 2.)
    np.testing.assert_allclose(image[-1] - image[0], np.tile([.08, 0.], (133, 1)), atol=1e-6)
    assert (image[:, 5, 0] > image[:, 6, 0]).all()
    assert (image[:, 0, 1] < image[:, 5, 1]).all()
    plot_frames(image, np.ones((t, 133)), args.output / "synthetic_projection.png")
    report = dict(data="synthetic kinematic fixture", frames=t, landmarks=133, fk=True,
                  left_right=True, y_down=True, chunked_fk=True, masked_face=True)
    (args.output / "geometry_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
