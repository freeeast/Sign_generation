"""Plot the paired pilot outcomes from evaluate_feedback CSVs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "mismatched_motion": "Mismatched motion",
    "static": "Static pose",
    "coordinates_noise": "All-point noise",
    "frozen_handshape": "Frozen handshape",
    "time_reverse": "Time reversal",
    "time_scramble": "Frame shuffle",
    "confidence_one_for_visible": "Visible scores set to 1",
    "similarity_control": "Scale + shift control",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["role"] != "known_output_positive_control"]
    by_clip = sorted(set(r["clip_id"] for r in rows))
    doses = (0.025, 0.05, 0.10, 0.20)
    values = {(r["clip_id"], r["variant"]): float(r["delta_nll"]) for r in rows}
    curves = np.array([[values[(clip, f"hand_noise_{dose:g}")] for dose in doses] for clip in by_clip])
    fig, (dose_ax, cmp_ax) = plt.subplots(1, 2, figsize=(11, 4.9), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.subplots_adjust(left=.08, right=.98, bottom=.18, top=.84, wspace=.74)
    fig.suptitle("Uni-Sign feedback on public How2Sign poses", fontsize=15, weight="bold", y=.95)
    dose_ax.axhline(0., color="#657487", lw=1, ls="--")
    for curve in curves:
        dose_ax.plot(np.array(doses)*100, curve, color="#7999ae", alpha=.55, lw=1)
    dose_ax.plot(np.array(doses)*100, np.median(curves, axis=0), color="#195778", lw=2.5,
                 marker="o", label="Clip median")
    dose_ax.set(xlabel="Finger noise SD (% of shoulder width)", ylabel="Change in target NLL (nats/token)",
                title="Finger-coordinate noise")
    dose_ax.set_xticks(np.array(doses)*100)
    dose_ax.legend(frameon=False, loc="upper left")
    cmp_ax.axvline(0., color="#657487", lw=1, ls="--")
    for index, variant in enumerate(LABELS):
        delta = [values[(clip, variant)] for clip in by_clip]
        cmp_ax.scatter(delta, np.full(len(delta), index), c="#7999ae", alpha=.64, s=26)
        cmp_ax.scatter(np.median(delta), index, c="#195778", s=31, marker="D", zorder=3)
    cmp_ax.set_yticks(range(len(LABELS)), LABELS.values())
    cmp_ax.invert_yaxis()
    cmp_ax.set(xlabel="Change in target NLL (nats/token)", title="Paired controls")
    for ax in (dose_ax, cmp_ax):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.text(.08, .04, "8 clips. Positive values indicate lower likelihood for the annotated sentence.",
             color="#394859", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=185)
    plt.close(fig)


if __name__ == "__main__":
    main()
