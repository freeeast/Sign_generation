"""Score exact How2Sign inputs and their paired virtual variants with the existing Uni-Sign adapter."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .evaluate_feedback import write_csv
from .pose import load_pose
from .project_keypoints import plot_frames
from .unisign import UniSign, cosine
from .virtual_pose import model_sha, virtual_inputs


def load_manifest(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows or len({r["clip_id"] for r in rows}) != len(rows) or {r["split"] for r in rows} != {"val"}:
        raise ValueError("Use a nonempty, unique validation split manifest")
    return rows


def observations(root, row):
    path = root / (row["clip_id"] + ".npz")
    xy, scores = load_pose(path)
    with np.load(path, allow_pickle=False) as info:
        wh = int(info["width"]), int(info["height"])
    if len(xy) != row["rgb_frames"]:
        raise ValueError(f"{row['clip_id']}: observed frame count differs from manifest")
    return xy, scores, wh


def real_inputs(xy, score):
    mask_face = score.copy()
    mask_face[:, 23:91] = 0
    return dict(raw=(xy, score), visible_to_one=(xy, (score > .3).astype(np.float32)),
                face_mask_only=(xy, mask_face))


def score_clip(model, xy, scores, targets, decode, max_frames):
    # Reuse a single encoder unless the label bank would be expensive to decode as one batch.
    with model.torch.inference_mode():
        encoded, attention, embedding, frames = model._encode(xy, scores, max_frames)
        nll, token_loss = [], []
        for start in range(0, len(targets), 3):
            observed, per_token = model._score_targets(encoded, attention, targets[start:start + 3], token_nll=True)
            nll.extend(observed)
            token_loss.extend(per_token)
        prediction = None
        if decode:
            ids = model.model.mt5_model.generate(encoder_outputs=encoded, attention_mask=attention,
                                                max_new_tokens=100, num_beams=4)
            prediction = model.model.mt5_tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
    return dict(nll=nll, token_loss=token_loss, embedding=embedding, model_frames=frames, prediction=prediction)


def score_bank(model, rows, inputs, domain):
    ids = [r["clip_id"] for r in rows]
    lookup = {r["clip_id"]: r for r in rows}
    references = [r["reference"] for r in rows]
    scores = {}
    out, predictions = [], []
    for clip, variants in inputs.items():
        index = ids.index(clip)
        for name, (xy, confidence) in variants.items():
            primary = name in {"raw", "proxy_matched_constant1"}
            targets = references if primary else [references[index]]
            result = score_clip(model, xy, confidence, targets, primary, max_frames=256)
            scores[clip, name] = result
            if primary:
                out.extend(dict(domain=domain, pose_clip=clip, reference_clip=ref, own=clip == ref,
                                same_video=lookup[ref]["video_name"] == lookup[clip]["video_name"],
                                frame_gap=abs(lookup[ref]["rgb_frames"] - len(xy)), nll=nll)
                           for ref, nll in zip(ids, result["nll"]))
            predictions.append(dict(clip=clip, variant=name, frames_model=result["model_frames"],
                                    nll=result["nll"][index if primary else 0], prediction=result["prediction"]))
            print(domain, clip, name, round(predictions[-1]["nll"], 4), flush=True)
    return out, predictions, scores


def ranking(rows, bank, other, variant, *, query):
    # The input bank and query domain are explicit: virtual-to-real retrieval fixes each virtual query.
    names = [x["clip_id"] for x in rows]
    lookup = {x["clip_id"]: x for x in rows}
    scores = []
    for clip, actual_variant in bank:
        if actual_variant != variant:
            continue
        item = bank[clip, variant]
        for candidate in names:
            if (candidate, "raw") not in other:
                continue
            record = other[candidate, "raw"]
            expected = lookup[clip]
            target = names.index(clip)
            similarity = cosine(item["embedding"], record["embedding"])
            scores.append(dict(query_domain=query, query_clip=clip, candidate_clip=candidate,
                               match=clip == candidate,
                               same_video=expected["video_name"] == lookup[candidate]["video_name"],
                               frame_gap=abs(expected["rgb_frames"] - lookup[candidate]["rgb_frames"]),
                               candidate_nll_on_query_text=record["nll"][target], cosine=similarity))
    return scores



def load_real_cache(root, rows):
    """Recover the completed real-pose matrix and pooled embeddings by exact clip ID."""
    ids = [row["clip_id"] for row in rows]
    with (root / "real_only_nll_matrix.csv").open(newline="", encoding="utf-8") as handle:
        nll = {(row["pose_clip"], row["reference_clip"]): float(row["nll"])
               for row in csv.DictReader(handle)}
    if not all((pose, reference) in nll for pose in ids for reference in ids):
        raise ValueError("Real NLL cache does not cover all manifest IDs")
    with np.load(root / "real_only_embeddings.npz", allow_pickle=False) as embeddings:
        return {(clip, "raw"): {"nll": [nll[clip, ref] for ref in ids], "embedding": embeddings[clip]}
                for clip in ids}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("repo", "checkpoint", "mt5", "manifest", "pose-dir", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    for flag in ("model-dir", "motion-dir", "camera-dir"):
        parser.add_argument(f"--{flag}", type=Path)
    parser.add_argument("--hand-mean", choices=("model", "flat"), default="model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reuse-real", type=Path, help="Reuse E3 real-only NLL and embeddings from this directory")
    parser.add_argument("--probe", action="store_true", help="Run candidate perturbations when genuine FK is available")
    parser.add_argument("--probe-only", action="store_true",
                        help="Use the saved E1_real geometry to run E3.5 without replaying E3")
    args = parser.parse_args()
    if (args.probe or args.probe_only) and args.model_dir is None:
        parser.error("--probe requires the neutral SMPL-X model and paired motions")
    if args.model_dir is not None and args.motion_dir is None:
        parser.error("--model-dir requires --motion-dir")
    results = args.output
    for name in ("E1_real", "E3", "E3_5"):
        (results / name).mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest)
    real_data = {row["clip_id"]: observations(args.pose_dir, row) for row in rows}
    model = UniSign(args.repo, args.checkpoint, args.mt5, args.device, args.threads)
    real = {key: real_inputs(xy, score) for key, (xy, score, _) in real_data.items()}
    if args.reuse_real:
        real_scores = load_real_cache(args.reuse_real, rows)
    else:
        cross, real_outputs, real_scores = score_bank(model, rows, real, "real")
        write_csv(results / "E3/real_only_nll_matrix.csv", cross)
        write_csv(results / "E3/real_only_conditions.csv", real_outputs)
        write_csv(results / "E3/real_only_retrieval.csv", ranking(rows, real_scores, real_scores, "raw", query="real"))
        np.savez_compressed(results / "E3/real_only_embeddings.npz",
                            **{key: val["embedding"] for (key, mode), val in real_scores.items() if mode == "raw"})
    if args.model_dir is None:
        return

    asset_hash = model_sha(args.model_dir)
    if args.probe_only:
        from .probe_3d import probe

        previous = json.loads((results / "E1_real/geometry_qc.json").read_text())
        if (previous["asset_sha256"], previous["hand_mean"]) != (asset_hash, args.hand_mean):
            raise ValueError("Saved geometry used a different SMPL-X asset or hand mean")
        for row in rows:
            clip = row["clip_id"]
            if not row["fit_frames"]:
                continue
            observed, score, wh = real_data[clip]
            path = args.motion_dir / (clip + ".npz")
            with np.load(results / "E1_real" / clip / "fk_and_projection.npz", allow_pickle=False) as geometry:
                probe(model, args.model_dir, args.hand_mean, args.device, row, path,
                      geometry["landmarks3d"], geometry["camera"], observed, score, wh,
                      real_scores[clip, "raw"]["embedding"], results / "E3_5")
        return
    virtual, geometry = {}, []
    for row in rows:
        clip = row["clip_id"]
        if not row["fit_frames"]:
            continue
        if not row["alignment_candidate"]:
            raise ValueError(f"Review source provenance and alignment before processing {clip}")
        observed, score, wh = real_data[clip]
        path = args.motion_dir / (clip + ".npz")
        affine = None
        if args.camera_dir is not None:
            affine = np.load(args.camera_dir / (clip + ".npy"), allow_pickle=False)
        points, variants, camera, quality = virtual_inputs(
            path, args.model_dir, observed, score, wh, args.hand_mean, args.device, affine)
        virtual[clip] = variants
        geometry.append(dict(clip=clip, **quality))
        target = results / "E1_real" / clip
        target.mkdir(exist_ok=True)
        np.savez_compressed(target / "fk_and_projection.npz", landmarks3d=points,
                            camera=camera, keypoints=variants["proxy_matched_constant1"][0])
        plot_frames(*variants["proxy_matched_constant1"], target / "virtual_projection.png")
        plot_3d(points, target / "fitted_motion_3d.png")
        if args.probe:
            from .probe_3d import probe
            probe(model, args.model_dir, args.hand_mean, args.device, row, path, points, camera,
                  observed, score, wh, real_scores[clip, "raw"]["embedding"], results / "E3_5")
    (results / "E1_real/geometry_qc.json").write_text(json.dumps(dict(asset_sha256=asset_hash,
        hand_mean=args.hand_mean, rows=geometry), indent=2) + "\n")
    matrix, outputs, v_scores = score_bank(model, rows, virtual, "virtual")
    write_csv(results / "E3/virtual_conditions.csv", outputs)
    write_csv(results / "E3/virtual_nll_matrix.csv", matrix)
    write_csv(results / "E3/virtual_retrieval.csv",
              ranking(rows, v_scores, real_scores, "proxy_matched_constant1", query="virtual"))
    comparisons = []
    for query in virtual:
        index = next(i for i, row in enumerate(rows) if row["clip_id"] == query)
        base = real_scores[query, "raw"]
        target = rows[index]["reference"]
        a_full = model.teacher_log_probs(*real[query]["raw"], target, 256)
        b_full = model.teacher_log_probs(*virtual[query]["proxy_matched_constant1"], target, 256)
        mix = model.torch.logaddexp(a_full, b_full) - np.log(2.)
        js = ((a_full.exp() * (a_full - mix)).sum(-1) + (b_full.exp() * (b_full - mix)).sum(-1)) / 2
        js_mean = float(js.mean().clamp_min(0))
        top1_agreement = float((a_full.argmax(-1) == b_full.argmax(-1)).float().mean())
        label_ids = model.model.mt5_tokenizer(
            [target], return_tensors="pt", truncation=True, max_length=50)["input_ids"][0]
        true_token_probs = a_full[model.torch.arange(len(label_ids)), label_ids].exp().numpy()
        for name in virtual[query]:
            candidate = v_scores[query, name]
            offset = index if name == "proxy_matched_constant1" else 0
            b = np.asarray(candidate["token_loss"][offset])
            comparisons.append(dict(clip=query, variant=name, real_nll=base["nll"][index],
                virtual_nll=candidate["nll"][offset], cosine=cosine(base["embedding"], candidate["embedding"]),
                mean_abs_reference_token_probability_gap=float(np.abs(true_token_probs - np.exp(-b)).mean()),
                full_vocabulary_js_nats=js_mean if name == "proxy_matched_constant1" else None,
                full_vocabulary_top1_agreement=top1_agreement if name == "proxy_matched_constant1" else None))
    write_csv(results / "E3/domain_gap.csv", comparisons)


def plot_3d(points, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 3), layout="constrained")
    limits = points[:, np.r_[0, 5:11, 91:133]].reshape(-1, 3)
    center = (limits.max(axis=0) + limits.min(axis=0)) / 2
    radius = np.ptp(limits, axis=0).max() / 2
    for i, n in enumerate(np.linspace(0, len(points) - 1, 4, dtype=int), 1):
        ax = fig.add_subplot(1, 4, i, projection="3d")
        ax.view_init(elev=12, azim=-70)
        for links, color in (([[0, 5, 7, 9], [0, 6, 8, 10], [5, 6]], "0.3"),
                             ([[91, *range(a, a + 4)] for a in range(92, 112, 4)], "#177d9b"),
                             ([[112, *range(a, a + 4)] for a in range(113, 133, 4)], "#d36132")):
            for link in links:
                x, y, z = points[n, link].T
                ax.plot(x, z, -y, marker=".", markersize=2, color=color)
        ax.set(xlim=(center[0] - radius, center[0] + radius),
               ylim=(center[2] - radius, center[2] + radius),
               zlim=(-center[1] - radius, -center[1] + radius), title=f"Frame {n}")
        ax.set_box_aspect((1, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
