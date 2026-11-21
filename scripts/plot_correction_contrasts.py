"""Plot local hand restoration outcomes against held-out handshape error."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    with (args.root / "correction_contrasts.csv").open(newline="", encoding="utf-8") as source:
        rows = [r for r in csv.DictReader(source) if r["control"] == "corrupted16"]
    clips = {"_2FBDaOPYig": "A", "0zvsqf23tmw": "B", "3TrMyzNWGpY": "C",
             "bBHYQCDp_9A": "D", "f5EGPzGSCJs": "E"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
    for ax, label, tag in ((axes[0], "Finger articulation", "finger_articulation"),
                           (axes[1], "Wrist orientation", "wrist_orientation")):
        subset = [r for r in rows if r["family"] == tag]
        ax.axhline(0, c=".5", lw=1)
        for r in subset:
            horizontal = float(r["restored_minus_control_geometry_B_px"])
            vertical = float(r["restored_minus_control_nll"])
            name = next(short for source, short in clips.items() if r["clip"].startswith(source))
            ax.scatter(horizontal, vertical, c="#c24f37" if vertical > 0 else "#1e6f84", s=40)
            ax.annotate(name, (horizontal, vertical), xytext=(5, 3), textcoords="offset points")
        ax.set(title=label, xlabel="Δ held-out handshape error (px)", ylabel="Δ NLL (nats/token)",
               xlim=(-4.5, .05), ylim=(-.08, .065))
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("8° restoration versus 16° corruption · negative values favor restoration")
    fig.savefig(args.root / "hand_restoration_conflicts.png", dpi=170)


if __name__ == "__main__":
    main()
