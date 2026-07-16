from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def _validate_score_arrays(target_scores, non_target_scores):
    target_scores = np.asarray(target_scores, dtype=float)
    non_target_scores = np.asarray(non_target_scores, dtype=float)

    if target_scores.size == 0:
        raise ValueError("target_scores must contain at least one value")
    if non_target_scores.size == 0:
        raise ValueError("non_target_scores must contain at least one value")

    return target_scores, non_target_scores


def compute_eer_details(target_scores, non_target_scores):
    target_scores, non_target_scores = _validate_score_arrays(
        target_scores, non_target_scores
    )

    labels = np.concatenate(
        [
            np.ones(target_scores.size, dtype=int),
            np.zeros(non_target_scores.size, dtype=int),
        ]
    )
    scores = np.concatenate([target_scores, non_target_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    if thresholds.size and thresholds[0] > np.max(scores):
        fpr = fpr[1:]
        fnr = fnr[1:]
        thresholds = thresholds[1:]

    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])

    return {
        "eer": eer,
        "threshold": threshold,
        "thresholds": thresholds,
        "fpr": fpr,
        "fnr": fnr,
        "target_scores": target_scores,
        "non_target_scores": non_target_scores,
    }


def compute_eer(target_scores, non_target_scores):
    return compute_eer_details(target_scores, non_target_scores)["eer"]


def _configure_matplotlib():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.figsize": (3.5, 2.5),
            "figure.dpi": 300,
        }
    )


def _score_xlim(target_scores, non_target_scores):
    all_scores = np.concatenate([target_scores, non_target_scores])
    score_min = float(np.min(all_scores))
    score_max = float(np.max(all_scores))
    if score_min == score_max:
        padding = 0.05 if score_min == 0 else abs(score_min) * 0.05
        return (score_min - padding, score_max + padding)

    padding = 0.03 * (score_max - score_min)
    return (score_min - padding, score_max + padding)


def plot_eer_histogram(target_scores, non_target_scores, out_dir, filename_base):
    details = compute_eer_details(target_scores, non_target_scores)
    target_scores = details["target_scores"]
    non_target_scores = details["non_target_scores"]
    eer = details["eer"]
    threshold = details["threshold"]
    thresholds = details["thresholds"]
    fpr = details["fpr"]
    fnr = details["fnr"]

    _configure_matplotlib()

    fig, ax1 = plt.subplots()

    weights_non_target = np.ones_like(non_target_scores) / len(non_target_scores)
    weights_target = np.ones_like(target_scores) / len(target_scores)

    ax1.hist(
        non_target_scores,
        bins=50,
        weights=weights_non_target,
        alpha=0.4,
        color="blue",
        label="Non-mated",
    )
    ax1.hist(
        target_scores,
        bins=50,
        weights=weights_target,
        alpha=0.4,
        color="green",
        label="Mated",
    )
    ax1.set_xlabel("Score ($s$)")
    ax1.set_ylabel("Relative Frequency")
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.plot(
        thresholds,
        fpr,
        color="blue",
        linewidth=1.5,
        label="False Acceptance Rate",
    )
    ax2.plot(
        thresholds,
        fnr,
        color="green",
        linewidth=1.5,
        label="False Reject Rate",
    )
    ax2.set_ylabel("Error Rate")
    ax2.set_ylim([0.0, 1.0])

    ax2.plot(threshold, eer, marker="o", color="red", markersize=4, zorder=5)

    xlims = _score_xlim(target_scores, non_target_scores)
    ax1.set_xlim(xlims)

    ax2.hlines(
        eer,
        xmin=threshold,
        xmax=xlims[1],
        colors="red",
        linestyles="dotted",
        linewidth=1.5,
    )
    ax2.vlines(
        threshold,
        ymin=0,
        ymax=eer,
        colors="red",
        linestyles="dotted",
        linewidth=1.5,
    )
    ax2.text(
        threshold,
        min(eer + 0.05, 0.98),
        f"EER = {eer * 100:.2f}%",
        color="red",
        fontweight="bold",
        fontsize=8,
        ha="center",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
        ),
    )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ordered_handles = [handles2[1], handles2[0], handles1[1], handles1[0]]
    ordered_labels = [labels2[1], labels2[0], labels1[1], labels1[0]]
    ax1.legend(
        ordered_handles,
        ordered_labels,
        loc="upper left",
        framealpha=0.9,
        borderaxespad=0.2,
        borderpad=0.4,
        labelspacing=0.4,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{filename_base}.pdf"
    png_path = out_dir / f"{filename_base}.png"

    plt.tight_layout()
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "eer": eer,
        "threshold": threshold,
        "pdf_path": pdf_path,
        "png_path": png_path,
    }
