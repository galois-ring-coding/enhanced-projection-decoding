"""Render the manuscript figure from the standard-rejection Lee scale summary."""

import csv
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
SUMMARY = (
    ROOT
    / "results"
    / "lee_scale_rejection_20261119"
    / "analysis"
    / "panel_summary.csv"
)
METHODS = (
    ("ep_reject", "CCP\u2013Prange", "#1670a5", "o"),
    ("proj_binary_t", "Matched binary\nbaseline", "#ba3e48", "s"),
)


def main():
    with SUMMARY.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    rows = [row for row in records if row["group"] == "sphere"]
    lengths = sorted({int(row["n"]) for row in rows})
    assert lengths == [48, 64, 96]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(5.0, 5.45), sharex=True)
    for index, n in enumerate(lengths):
        for method, label, color, marker in METHODS:
            data = sorted(
                (
                    row
                    for row in rows
                    if int(row["n"]) == n and row["method"] == method
                ),
                key=lambda row: float(row["extra_fraction"]),
            )
            assert len(data) == int(data[0]["K"]) + 1
            x = np.array([float(row["extra_fraction"]) for row in data])
            axes[index, 0].plot(
                x,
                [float(row["coverage_first_trials_lower_bound"]) for row in data],
                "--",
                color=color,
                linewidth=0.9,
            )
            for column, metric, scale in (
                (0, "first_trials", 1),
                (1, "online_ns", 1e6),
            ):
                mean = np.array([float(row[metric]) / scale for row in data])
                low = np.array(
                    [float(row[metric + "_low95"]) / scale for row in data]
                )
                high = np.array(
                    [float(row[metric + "_high95"]) / scale for row in data]
                )
                assert np.all(low <= mean) and np.all(mean <= high)
                axes[index, column].errorbar(
                    x,
                    mean,
                    yerr=np.array([mean - low, high - mean]),
                    fmt=marker,
                    color=color,
                    markersize=2.6,
                    markeredgewidth=0.5,
                    capsize=1.4,
                    elinewidth=0.6,
                    errorevery=(
                        len(x)
                        if column == 0 and method == "proj_binary_t"
                        else 1
                    ),
                )
                axes[index, column].plot(
                    x, mean, color=color, linewidth=0.75
                )
        k, t = int(data[0]["K"]), int(data[0]["t"])
        for column in range(2):
            ax = axes[index, column]
            ax.set_title(f"$n={n}, K={k}, t={t}$", pad=4)
            ax.set_xlim(-0.04, 1.04)
            ax.set_xticks([0, 0.5, 1])
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", color="0.9", linewidth=0.5)
        axes[index, 0].set_ylabel("Mean selections")
        axes[index, 1].set_ylabel("Online time (ms)")
    for ax in axes[-1]:
        ax.set_xlabel(r"$(K-k_0)/K$")
    handles = [
        Line2D(
            [0], [0], color=color, marker=marker, markersize=3, label=label
        )
        for _, label, color, marker in METHODS
    ]
    handles.append(
        Line2D(
            [0], [0], color="0.4", linestyle="--", label="Coverage lower bound"
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1.6,
        columnspacing=1.1,
    )
    fig.subplots_adjust(
        left=0.115, right=0.985, bottom=0.085, top=0.91, hspace=0.50, wspace=0.38
    )
    output = ROOT / "figures"
    output.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"lee_scale_ccp.{suffix}", dpi=220)
    plt.close(fig)
    print("Summary SHA-256:", hashlib.sha256(SUMMARY.read_bytes()).hexdigest())
    print("Wrote", output / "lee_scale_ccp.pdf")


if __name__ == "__main__":
    main()
