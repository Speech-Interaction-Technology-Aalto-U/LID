from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


LOG_2 = np.log(2.0)


def _labels_and_scores(dataframe: pd.DataFrame):
    if "score" not in dataframe.columns:
        raise ValueError("Expected a `score` column in the evaluation dataframe.")

    scores = pd.to_numeric(dataframe["score"], errors="coerce")
    if "mated" in dataframe.columns:
        labels = pd.to_numeric(dataframe["mated"], errors="coerce")
    elif {"enroll_spk", "trial_spk"}.issubset(dataframe.columns):
        labels = (
            dataframe["enroll_spk"].astype(str) == dataframe["trial_spk"].astype(str)
        ).astype(int)
    else:
        raise ValueError(
            "Expected either a `mated` column or both `enroll_spk` and `trial_spk`."
        )

    valid = scores.notna() & labels.notna()
    cleaned_scores = scores.loc[valid].astype(float)
    cleaned_labels = (labels.loc[valid].astype(float) > 0).astype(int)

    if cleaned_scores.empty:
        raise ValueError("No valid rows were available for statistical analysis.")
    if cleaned_labels.nunique() < 2:
        raise ValueError("Statistical analysis requires both mated and non-mated rows.")

    label_values = cleaned_labels.to_numpy(dtype=int)
    score_values = cleaned_scores.to_numpy(dtype=float)
    return (
        label_values,
        score_values,
        score_values[label_values == 1],
        score_values[label_values == 0],
    )


def _prior_log_odds(labels):
    mated_count = int(labels.sum())
    non_mated_count = int((labels == 0).sum())
    if mated_count == 0 or non_mated_count == 0:
        raise ValueError("Expected both mated and non-mated rows.")
    return float(np.log(mated_count / non_mated_count))


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


def _x_limits(scores):
    minimum = float(np.min(scores))
    maximum = float(np.max(scores))
    span = maximum - minimum
    padding = span * 0.05 if span > 0 else max(abs(minimum), 1.0) * 0.05
    return [minimum - padding, maximum + padding]


def _fit_isotonic(dataframe: pd.DataFrame):
    labels, scores, mated_scores, non_mated_scores = _labels_and_scores(dataframe)
    ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
    ir.fit(scores, labels)
    return ir, labels, scores, mated_scores, non_mated_scores


def plot_pav_calibration(dataframe: pd.DataFrame, out_dir, filename_base="pav_calibration"):
    ir, _labels, scores, mated_scores, non_mated_scores = _fit_isotonic(dataframe)

    _configure_matplotlib()
    fig, ax1 = plt.subplots()

    weights_non_mated = np.ones_like(non_mated_scores) / len(non_mated_scores)
    weights_mated = np.ones_like(mated_scores) / len(mated_scores)

    ax1.hist(
        non_mated_scores,
        bins=50,
        weights=weights_non_mated,
        alpha=0.4,
        color="blue",
        label="Non-mated",
    )
    ax1.hist(
        mated_scores,
        bins=50,
        weights=weights_mated,
        alpha=0.4,
        color="green",
        label="Mated",
    )
    ax1.set_xlabel("Score ($s$)")
    ax1.set_ylabel("Relative Frequency")
    ax1.set_ylim(bottom=0)

    x_plot = np.linspace(scores.min(), scores.max(), 1000)
    y_plot = ir.predict(x_plot)

    ax2 = ax1.twinx()
    ax2.plot(
        x_plot,
        y_plot,
        color="red",
        linewidth=1.8,
        label=r"PAV Probability $P(Mated \mid s)$",
    )
    ax2.set_ylabel(r"Posterior Probability $P(Mated \mid s)$")
    ax2.set_ylim([0.0, 1.05])
    ax1.set_xlim(_x_limits(scores))

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
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
        "pdf_path": pdf_path,
        "png_path": png_path,
    }


def compute_cllr(dataframe: pd.DataFrame):
    ir, labels, _scores, mated_scores, non_mated_scores = _fit_isotonic(dataframe)
    prior_log_odds = _prior_log_odds(labels)

    p_mated_given_s_mated = ir.predict(mated_scores)
    p_mated_given_s_non_mated = ir.predict(non_mated_scores)
    epsilon = 1e-9
    p_mated_clipped = np.clip(p_mated_given_s_mated, epsilon, 1 - epsilon)
    p_non_mated_clipped = np.clip(p_mated_given_s_non_mated, epsilon, 1 - epsilon)

    post_log_odds_mated = np.log(p_mated_clipped / (1 - p_mated_clipped))
    post_log_odds_non_mated = np.log(p_non_mated_clipped / (1 - p_non_mated_clipped))
    llr_mated = post_log_odds_mated - prior_log_odds
    llr_non_mated = post_log_odds_non_mated - prior_log_odds

    cost_mated = np.logaddexp(0.0, -llr_mated) / LOG_2
    cost_non_mated = np.logaddexp(0.0, llr_non_mated) / LOG_2

    return float(0.5 * (np.mean(cost_mated) + np.mean(cost_non_mated)))
