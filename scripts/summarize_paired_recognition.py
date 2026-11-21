"""Summarize paired E3 rankings and preprocessing controls."""
import argparse
import csv
from itertools import permutations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluate_feedback import write_csv


def load(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def permutation_p(matrix):
    """Exact one-sided label-permutation check on the five-pair core."""
    observed = np.trace(matrix)
    scores = [matrix[np.arange(len(matrix)), perm].sum() for perm in permutations(range(len(matrix)))]
    return np.mean(np.asarray(scores) >= observed - 1e-12).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    matrix = load(root / "virtual_nll_matrix.csv")
    retrieval = load(root / "virtual_retrieval.csv")
    conditions = load(root / "domain_gap.csv")
    ids = list(dict.fromkeys(row["pose_clip"] for row in matrix))
    names = {clip: chr(65 + i) for i, clip in enumerate(ids)}
    scored = {(r["pose_clip"], r["reference_clip"]): float(r["nll"]) for r in matrix}
    similarities = {(r["query_clip"], r["candidate_clip"]): float(r["cosine"]) for r in retrieval}
    core_nll = np.array([[scored[pose, ref] for ref in ids] for pose in ids])
    core_similarity = np.array([[similarities[pose, ref] for ref in ids] for pose in ids])
    rows = []
    for clip in ids:
        real_pool = [r for r in retrieval if r["query_clip"] == clip]
        same = [r for r in real_pool if r["same_video"] == "True" and r["match"] == "False"]
        near = min(same or [r for r in real_pool if r["match"] == "False"],
                   key=lambda r: (int(r["frame_gap"]), r["candidate_clip"]))
        base = next(r for r in conditions if r["clip"] == clip and r["variant"] == "proxy_matched_constant1")
        diagonal = scored[clip, clip]
        negatives = [scored[other, clip] for other in ids if other != clip]
        sim = similarities[clip, clip]
        rows.append(dict(id=names[clip], clip=clip, real_nll=float(base["real_nll"]), virtual_nll=diagonal,
            nll_gap=diagonal-float(base["real_nll"]), nll_rank_5=1+sum(value < diagonal for value in negatives),
            nll_wins_4=sum(value > diagonal for value in negatives), nll_nearest_score_margin=min(negatives)-diagonal,
            same_pose_cosine=sim, retrieval_rank_9=1+sum(float(r["cosine"]) > sim for r in real_pool),
            nearest_real_negative=near["candidate_clip"], nearest_frame_gap=int(near["frame_gap"]),
            nearest_same_source=near["same_video"]=="True", nearest_cosine=float(near["cosine"]),
            same_source_n=len(same), same_source_cosine_wins=sum(sim > float(r["cosine"]) for r in same),
            js_nats=float(base["full_vocabulary_js_nats"]), top1_token_agreement=float(base["full_vocabulary_top1_agreement"])))
    write_csv(root / "paired_summary.csv", rows)
    primary = {(r["clip"]): float(r["virtual_nll"]) for r in conditions if r["variant"] == "proxy_matched_constant1"}
    variants = list(dict.fromkeys(r["variant"] for r in conditions if r["variant"] != "proxy_matched_constant1"))
    abl = []
    for variant in variants:
        deltas = np.array([float(r["virtual_nll"])-primary[r["clip"]] for r in conditions if r["variant"] == variant])
        abl.append(dict(variant=variant, median_delta_nll=np.median(deltas), min_delta_nll=deltas.min(),
                        max_delta_nll=deltas.max(), max_abs_delta_nll=np.abs(deltas).max()))
    write_csv(root / "paired_ablation_summary.csv", abl)
    out = dict(n_pairs=len(ids), retrieval_top1_of_9=sum(r["retrieval_rank_9"] == 1 for r in rows),
               nll_top1_of_5=sum(r["nll_rank_5"] == 1 for r in rows),
               nll_pairwise_wins=sum(r["nll_wins_4"] for r in rows),
               nll_pairwise_n=len(ids)*(len(ids)-1),
               same_source_cosine_wins=sum(r["same_source_cosine_wins"] for r in rows),
               same_source_n=sum(r["same_source_n"] for r in rows),
               median_nll_gap=float(np.median([r["nll_gap"] for r in rows])),
               median_matching_cosine=float(np.median(np.diag(core_similarity))),
               median_mismatching_cosine=float(np.median(core_similarity[~np.eye(len(ids), dtype=bool)])),
               median_js_nats=float(np.median([r["js_nats"] for r in rows])),
               median_token_agreement=float(np.median([r["top1_token_agreement"] for r in rows])),
               nll_label_permutation_p=permutation_p(-core_nll),
               cosine_label_permutation_p=permutation_p(core_similarity))
    (root / "paired_totals.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))

    margins = core_nll - np.diag(core_nll)[None, :]
    fig, axes = plt.subplots(1, 2, figsize=(10.9, 4.7), layout="constrained")
    lim = np.max(np.abs(margins))
    for ax, values, title, palette, mn, mx, colorbar, labels in (
        (axes[0], margins, "Virtual pose · fixed reference", "RdBu", -lim, lim, "Δ NLL vs matched pose", "Reference sentence"),
        (axes[1], core_similarity, "Virtual and real encoders", "viridis", -1, 1, "Cosine similarity", "Real pose"),
    ):
        image = ax.imshow(values, cmap=palette, vmin=mn, vmax=mx)
        ticks = list(names.values())
        ax.set(xticks=range(len(ids)), xticklabels=ticks, yticks=range(len(ids)), yticklabels=ticks,
               title=title, xlabel=labels, ylabel="Virtual pose")
        for i, row in enumerate(values):
            for j, v in enumerate(row):
                color = "white" if (ax is axes[0] and abs(v)/lim > .62) or (ax is axes[1] and v < .2) else "#11202b"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=9)
        fig.colorbar(image, ax=ax, fraction=.05, label=colorbar)
    fig.savefig(root / "paired_core_comparison.png", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    main()
