"""Controlled perturbations to recognition inputs; noise is shoulder-relative."""
import numpy as np

from .pose import BODY, FINGERS, LEFT, RIGHT


def shoulder_width(xy, scores):
    width = np.linalg.norm(xy[:, 5] - xy[:, 6], axis=-1)
    usable = (scores[:, [5, 6]] > 0.3).all(axis=-1) & (width > 1e-6)
    if not usable.any():
        raise ValueError("No confident shoulder pair to scale perturbations")
    return float(np.median(width[usable]))


def perturbations(xy, score, rng):
    shoulder = shoulder_width(xy, score)
    yield "clean", xy, score
    noise = rng.normal(size=xy[:, FINGERS].shape)
    valid_fingers = score[:, FINGERS] > 0.3
    for fraction in (0.025, 0.05, 0.10, 0.20):
        noisy = xy.copy()
        noisy[:, FINGERS] += noise * (shoulder * fraction) * valid_fingers[..., None]
        yield f"hand_noise_{fraction:g}", noisy, score

    frozen, frozen_scores = xy.copy(), score.copy()
    for hand in (LEFT, RIGHT):
        complete = (score[:, hand] > 0.3).all(axis=1)
        candidates = np.flatnonzero(complete)
        if len(candidates):
            anchor = candidates[len(candidates)//2]
        else:
            anchor = int(np.argmax((score[:, hand] > 0.3).sum(axis=1)))
        offsets = xy[anchor, hand] - xy[anchor, hand[0]]
        frozen[:, hand[1:]] = xy[:, hand[0], None] + offsets[1:]
        valid = ((score[:, hand[0], None] > 0.3) & (score[anchor, hand[1:]] > 0.3))
        frozen_scores[:, hand[1:]] *= valid
    yield "frozen_handshape", frozen, frozen_scores
    yield "time_reverse", xy[::-1].copy(), score[::-1].copy()
    order = rng.permutation(len(xy))
    yield "time_scramble", xy[order], score[order]
    best = int(np.argmax((score[:, np.r_[BODY, LEFT, RIGHT]] > 0.3).sum(axis=1)))
    yield "static", np.repeat(xy[best:best + 1], len(xy), axis=0), np.repeat(
        score[best:best + 1], len(xy), axis=0)
    yield "coordinates_noise", xy + rng.normal(size=xy.shape) * shoulder * 0.5, score

    # Stress the score channel without changing geometry or time.
    yield "confidence_one_for_visible", xy, np.where(score > 0.3, 1.0, score)
    # Negative control: an isotropic image-plane transform should cancel in crop_scale.
    yield "similarity_control", xy * 1.15 + np.array([0.05, -0.03], dtype=np.float32), score
