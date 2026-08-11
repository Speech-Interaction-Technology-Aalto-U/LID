"""
Create experiment plots and summary tables for local information disclosure results.

For each experiment, this script plots the test-trial probability and LID
distributions. Across experiments, it writes a combined LID CCDF, a summary
metrics table, calibration coefficients, and, when EER values are available, an
EER-vs-information-disclosure scatter plot.

Inputs:
    <experiment>/outputs/local_information_disclosures.csv
    <experiment>/outputs/calibration_parameters.json
    results/experiments/<experiment>/results.json
    results/experiments/<experiment>/alternative_metrics/results.json, optional

Outputs:
    results/experiments/<experiment>/plots/*.png and *.pdf
    results/summary/summary_table.csv
    results/summary/calibration_coeff.csv
    results/summary/lid_combined_ccdf.png and .pdf
    results/summary/eer_vs_infodisc_scatter.png and .pdf, when EER is available
"""

import json
from pathlib import Path
import shutil
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.utils import (
    PROJECT_ROOT,
    calibration_parameters_path,
    iter_experiment_dirs,
    lid_path,
    SEPARATOR,
    SEPARATOR2,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"
METRICS_FILE = "results.json"
ALTERNATIVE_METRICS_DIR_NAME = "alternative_metrics"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summary"
PAPER_FIGURES_DIR = PROJECT_ROOT / "results" / "paper_figures"


def configure_matplotlib():
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


def speaker_colors(n_speakers):
    if n_speakers == 0:
        return []

    colors = ["black"]
    if n_speakers > 1:
        color_positions = np.linspace(
            0.0,
            2.0,
            n_speakers - 1,
            endpoint=False,
        ) % 1.0
        colors.extend(plt.colormaps["gist_rainbow"](color_positions))
    return colors


def speaker_indicator_values(speaker_ids, speaker_metrics, column):
    if speaker_metrics is None:
        return None, None, None

    mean_column = f"trial_mean_{column}"
    aggregated_column = f"aggregated_{column}"
    summed_column = f"summed_{column}"
    required_columns = {
        "trial_spk",
        mean_column,
        aggregated_column,
        summed_column,
    }
    missing_columns = required_columns - set(speaker_metrics.columns)
    if missing_columns:
        raise ValueError(
            f"Speaker metrics is missing required columns: {sorted(missing_columns)}"
        )

    metrics = speaker_metrics[list(required_columns)].copy()
    metrics["trial_spk"] = metrics["trial_spk"].astype(str)
    if metrics["trial_spk"].duplicated().any():
        duplicates = metrics.loc[
            metrics["trial_spk"].duplicated(keep=False),
            "trial_spk",
        ].head(10).tolist()
        raise ValueError(f"Duplicate speaker metrics found for: {duplicates}")

    metrics = metrics.set_index("trial_spk")
    speaker_keys = [str(speaker_id) for speaker_id in speaker_ids]
    missing_speakers = [key for key in speaker_keys if key not in metrics.index]
    if missing_speakers:
        raise ValueError(
            f"Speaker metrics is missing trial speakers: {missing_speakers[:10]}"
        )

    trial_means = metrics.loc[speaker_keys, mean_column].to_numpy(dtype=float)
    aggregated_values = metrics.loc[speaker_keys, aggregated_column].to_numpy(
        dtype=float
    )
    summed_values = metrics.loc[speaker_keys, summed_column].to_numpy(dtype=float)
    if not (
        np.isfinite(trial_means).all()
        and np.isfinite(aggregated_values).all()
        and np.isfinite(summed_values).all()
    ):
        raise ValueError(f"Speaker metrics contains non-finite {column} values")
    return trial_means, aggregated_values, summed_values


def _format_boundary_value(value):
    return f"{value:.6g}"


def plot_summed_evidence(
    ax,
    values,
    y_values,
    x_min,
    x_max,
    zorder,
    boundary_values=(),
    boundary_epsilon=0.0,
):
    """Plot summed evidence without allowing extreme values to expand the axis."""
    values = np.asarray(values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    x_range = x_max - x_min
    inset = max(x_range * 0.008, np.finfo(float).eps)
    left = values < x_min
    right = values > x_max
    exact_boundary = np.zeros(len(values), dtype=bool)
    for boundary_value in boundary_values:
        exact_boundary |= np.isclose(
            values,
            boundary_value,
            rtol=1e-9,
            atol=max(boundary_epsilon, 1e-12),
        )
    exact_boundary &= left | right
    left &= ~exact_boundary
    right &= ~exact_boundary
    inside = ~(left | right)
    label = "Longitudinal summed evidence"
    label_used = False

    for selected, marker, marker_x in (
        (
            inside,
            "D",
            np.clip(values, x_min + inset, x_max - inset),
        ),
        (left, "<", np.full_like(values, x_min + inset)),
        (right, ">", np.full_like(values, x_max - inset)),
    ):
        if not selected.any():
            continue
        ax.scatter(
            marker_x[selected],
            y_values[selected],
            marker=marker,
            s=34,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            label=label if not label_used else None,
            zorder=zorder,
        )
        label_used = True

    for index in np.flatnonzero(left | right):
        is_left = bool(left[index])
        ax.annotate(
            _format_boundary_value(values[index]),
            xy=(x_min + inset if is_left else x_max - inset, y_values[index]),
            xytext=(5 if is_left else -5, 0),
            textcoords="offset points",
            ha="left" if is_left else "right",
            va="center",
            fontsize=5.5,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "pad": 0.4,
            },
            zorder=zorder + 1,
        )


def plot_metric(
    df,
    column,
    title,
    xlabel,
    out_dir,
    filename_base,
    baseline_val=None,
    baseline_label=None,
    loc="upper right",
):
    valid_rows = df.loc[np.isfinite(df[column]), ["trial_spk", column]]
    if valid_rows.empty:
        raise ValueError(f"No finite values found in column: {column}")

    fig, ax = plt.subplots()

    speaker_scores = [
        speaker_rows[column].to_numpy()
        for _, speaker_rows in valid_rows.groupby("trial_spk", sort=False)
    ]
    speaker_weights = [
        np.full(len(scores), 1.0 / len(valid_rows))
        for scores in speaker_scores
    ]

    ax.hist(
        speaker_scores,
        bins=50,
        weights=speaker_weights,
        stacked=True,
        alpha=0.4,
        color=speaker_colors(len(speaker_scores)),
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Relative Frequency")
    ax.set_ylim(bottom=0)

    if baseline_val is not None:
        ax.axvline(
            x=baseline_val,
            color="red",
            linestyle="dotted",
            linewidth=1.5,
            label=baseline_label,
        )
        ax.legend(
            loc=loc,
            framealpha=0.9,
            borderaxespad=0.2,
            borderpad=0.4,
            labelspacing=0.4,
        )

    plt.tight_layout()

    png_path = out_dir / f"{filename_base}.png"
    pdf_path = out_dir / f"{filename_base}.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return [png_path, pdf_path]


def plot_metric_by_speaker(
    df,
    column,
    title,
    xlabel,
    out_dir,
    filename_base,
    baseline_val=None,
    baseline_label=None,
    loc="upper right",
    speaker_metrics=None,
    summed_boundary_values=(),
    summed_boundary_epsilon=0.0,
    display_min=None,
    display_max=None,
    dpi=300,
):
    valid_rows = df.loc[np.isfinite(df[column]), ["trial_spk", column]]
    if valid_rows.empty:
        raise ValueError(f"No finite values found in column: {column}")

    speaker_groups = list(valid_rows.groupby("trial_spk", sort=False))
    speaker_ids = [speaker_id for speaker_id, _ in speaker_groups]
    trial_means, aggregated_values, summed_values = speaker_indicator_values(
        speaker_ids,
        speaker_metrics,
        column,
    )
    all_scores = valid_rows[column].to_numpy()
    plotted_values = [all_scores]
    if trial_means is not None:
        plotted_values.extend((trial_means, aggregated_values))
    plotted_values = np.concatenate(plotted_values)
    score_min = float(plotted_values.min())
    score_max = float(plotted_values.max())
    score_range = score_max - score_min
    if score_range == 0.0:
        score_range = max(abs(score_min) * 0.1, 1.0)
    padding = score_range * 0.04
    x_min = score_min - padding
    x_max = score_max + padding
    if column == "p":
        x_min = max(0.0, x_min)
        x_max = min(1.0, x_max)
    if display_min is not None:
        x_min = float(display_min)
    if display_max is not None:
        x_max = float(display_max)
    x_grid = np.linspace(x_min, x_max, 400)

    figure_height = max(4.0, 0.18 * len(speaker_ids) + 1.5)
    fig, ax = plt.subplots(figsize=(7.0, figure_height))
    ridge_colors = speaker_colors(len(speaker_ids))
    ridge_baselines = np.arange(len(speaker_ids) - 1, -1, -1, dtype=float)

    for index, (_, speaker_rows) in enumerate(speaker_groups):
        scores = speaker_rows[column].to_numpy()
        # Silverman's rule with an IQR-based scale keeps small speaker groups smooth.
        score_std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        q1, q3 = np.percentile(scores, [25, 75])
        robust_scale = min(score_std, float(q3 - q1) / 1.34)
        if robust_scale <= 0.0:
            robust_scale = score_std
        minimum_bandwidth = (x_max - x_min) / 200.0
        bandwidth = max(
            0.9 * robust_scale * len(scores) ** (-0.2),
            minimum_bandwidth,
        )
        standardized_distances = (
            x_grid[:, np.newaxis] - scores[np.newaxis, :]
        ) / bandwidth
        density = np.exp(-0.5 * standardized_distances**2).sum(axis=1)
        density /= len(scores) * bandwidth * np.sqrt(2.0 * np.pi)
        if density.max() > 0.0:
            density = density / density.max() * 0.82

        baseline = ridge_baselines[index]
        ax.fill_between(
            x_grid,
            baseline,
            baseline + density,
            facecolor=ridge_colors[index],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.65,
            zorder=2,
        )
        ax.hlines(
            baseline,
            x_min,
            x_max,
            color="black",
            linewidth=0.5,
            zorder=3,
        )

    if trial_means is not None:
        ax.scatter(
            trial_means,
            ridge_baselines + 0.2,
            marker="o",
            s=24,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            label="Direct per-speaker mean",
            zorder=5,
        )
        ax.scatter(
            aggregated_values,
            ridge_baselines + 0.48,
            marker="*",
            s=58,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            label="Longitudinal averaged evidence",
            zorder=6,
        )
        plot_summed_evidence(
            ax,
            summed_values,
            ridge_baselines + 0.68,
            x_min,
            x_max,
            zorder=7,
            boundary_values=summed_boundary_values,
            boundary_epsilon=summed_boundary_epsilon,
        )

    ax.set_yticks(
        ridge_baselines,
        labels=[
            f"{speaker_id} ({len(speaker_rows)} trials)"
            for speaker_id, speaker_rows in speaker_groups
        ],
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.25, len(speaker_ids) - 0.05)
    ax.set_title(title, pad=42 if trial_means is not None else None)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Trial Speaker")
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(axis="x", color="gray", alpha=0.2, linewidth=0.6)
    ax.set_axisbelow(True)

    if baseline_val is not None:
        ax.axvline(
            x=baseline_val,
            color="red",
            linestyle="dotted",
            linewidth=1.5,
            label=baseline_label,
        )
    if baseline_val is not None or trial_means is not None:
        legend_options = {
            "framealpha": 0.9,
            "borderaxespad": 0.2,
            "borderpad": 0.4,
            "labelspacing": 0.4,
        }
        if trial_means is not None:
            legend_options.update(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=2,
            )
        else:
            legend_options["loc"] = loc
        ax.legend(**legend_options)

    plt.tight_layout()

    png_path = out_dir / f"{filename_base}.png"
    pdf_path = out_dir / f"{filename_base}.pdf"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return [png_path, pdf_path]


def plot_metric_by_speaker_heatmap(
    df,
    column,
    title,
    xlabel,
    out_dir,
    filename_base,
    baseline_val=None,
    baseline_label=None,
    loc="upper right",
    speaker_metrics=None,
    summed_boundary_values=(),
    summed_boundary_epsilon=0.0,
    display_min=None,
    display_max=None,
    dpi=300,
):
    valid_rows = df.loc[np.isfinite(df[column]), ["trial_spk", column]]
    if valid_rows.empty:
        raise ValueError(f"No finite values found in column: {column}")

    speaker_groups = list(valid_rows.groupby("trial_spk", sort=False))
    speaker_ids = [speaker_id for speaker_id, _ in speaker_groups]
    trial_means, aggregated_values, summed_values = speaker_indicator_values(
        speaker_ids,
        speaker_metrics,
        column,
    )
    bin_range_values = [valid_rows[column].to_numpy()]
    if trial_means is not None:
        bin_range_values.extend((trial_means, aggregated_values))
    bin_range_values = np.concatenate(bin_range_values)
    bin_min = float(bin_range_values.min())
    bin_max = float(bin_range_values.max())
    if display_min is not None:
        bin_min = min(bin_min, float(display_min))
    if display_max is not None:
        bin_max = max(bin_max, float(display_max))
    bin_edges = np.linspace(bin_min, bin_max, 51)
    bin_counts = np.array(
        [
            np.histogram(speaker_rows[column], bins=bin_edges)[0]
            for _, speaker_rows in speaker_groups
        ],
        dtype=float,
    )
    relative_frequencies = bin_counts / bin_counts.sum(axis=1, keepdims=True)

    figure_height = max(4.0, 0.18 * len(speaker_ids) + 1.5)
    fig, ax = plt.subplots(figsize=(7.0, figure_height))
    mesh = ax.pcolormesh(
        bin_edges,
        np.arange(len(speaker_ids) + 1),
        relative_frequencies,
        cmap="viridis",
        vmin=0.0,
        vmax=float(relative_frequencies.max()),
        shading="flat",
        rasterized=True,
    )

    ax.set_yticks(
        np.arange(len(speaker_ids)) + 0.5,
        labels=[
            f"{speaker_id} ({len(speaker_rows)} trials)"
            for speaker_id, speaker_rows in speaker_groups
        ],
    )
    ax.invert_yaxis()
    ax.set_title(title, pad=42 if trial_means is not None else None)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Trial Speaker")
    ax.tick_params(axis="y", labelsize=6)

    if trial_means is not None:
        y_values = np.arange(len(speaker_ids)) + 0.5
        ax.scatter(
            trial_means,
            y_values,
            marker="o",
            s=22,
            facecolor="white",
            edgecolor="black",
            linewidth=0.7,
            label="Direct per-speaker mean",
            zorder=4,
        )
        ax.scatter(
            aggregated_values,
            y_values + 0.12,
            marker="*",
            s=52,
            facecolor="white",
            edgecolor="black",
            linewidth=0.7,
            label="Longitudinal averaged evidence",
            zorder=5,
        )
        plot_summed_evidence(
            ax,
            summed_values,
            y_values - 0.15,
            float(bin_edges[0]),
            float(bin_edges[-1]),
            zorder=6,
            boundary_values=summed_boundary_values,
            boundary_epsilon=summed_boundary_epsilon,
        )

    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label("Within-Speaker Relative Frequency")

    if baseline_val is not None:
        ax.axvline(
            x=baseline_val,
            color="red",
            linestyle="dotted",
            linewidth=1.5,
            label=baseline_label,
        )
    if baseline_val is not None or trial_means is not None:
        legend_options = {
            "framealpha": 0.9,
            "borderaxespad": 0.2,
            "borderpad": 0.4,
            "labelspacing": 0.4,
        }
        if trial_means is not None:
            legend_options.update(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=2,
            )
        else:
            legend_options["loc"] = loc
        ax.legend(**legend_options)

    plt.tight_layout()

    png_path = out_dir / f"{filename_base}.png"
    pdf_path = out_dir / f"{filename_base}.pdf"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return [png_path, pdf_path]


def get_n_enrolments(experiment_dir, df_lid):
    metrics_path = experiment_dir / METRICS_FILE
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text())
        if "n_enrolments" in metrics:
            return int(metrics["n_enrolments"])

    n_vals = (2 ** df_lid["LID"]) / df_lid["p"]
    return int(np.round(np.median(n_vals)))


def load_lid_dataframe(experiment_dir):
    experiment_lid_path = lid_path(experiment_dir)
    if not experiment_lid_path.is_file():
        raise FileNotFoundError(f"Missing required LID file: {experiment_lid_path}")

    df_lid = pd.read_csv(experiment_lid_path)
    required_columns = {"trial_spk", "p", "LID"}
    missing_columns = required_columns - set(df_lid.columns)
    if missing_columns:
        raise ValueError(
            f"{experiment_lid_path} is missing required columns: {sorted(missing_columns)}"
        )

    return df_lid


def plot_probability_distribution(experiment_dir, df_lid, out_dir):
    n_enrolments = get_n_enrolments(experiment_dir, df_lid)

    return plot_metric(
        df_lid,
        column="p",
        title=f"Target Recognition Probability Distribution ({experiment_dir.name})",
        xlabel="Probability",
        out_dir=out_dir,
        filename_base="probability_distribution",
        baseline_val=1.0 / n_enrolments,
        baseline_label=f"Random Guess (p = 1/{n_enrolments})",
    )


def plot_probability_distribution_by_speaker(
    experiment_dir,
    df_lid,
    out_dir,
    speaker_metrics=None,
    dpi=300,
):
    n_enrolments = get_n_enrolments(experiment_dir, df_lid)

    return plot_metric_by_speaker(
        df_lid,
        column="p",
        title=f"Target Recognition Probability by Trial Speaker ({experiment_dir.name})",
        xlabel="Probability",
        out_dir=out_dir,
        filename_base="probability_distribution_by_speaker",
        baseline_val=1.0 / n_enrolments,
        baseline_label=f"Random Guess (p = 1/{n_enrolments})",
        speaker_metrics=speaker_metrics,
        summed_boundary_values=(1.0,),
        summed_boundary_epsilon=0.01,
        display_min=0.0,
        display_max=1.0,
        dpi=dpi,
    )


def plot_probability_distribution_by_speaker_heatmap(
    experiment_dir,
    df_lid,
    out_dir,
    speaker_metrics=None,
    dpi=300,
):
    n_enrolments = get_n_enrolments(experiment_dir, df_lid)

    return plot_metric_by_speaker_heatmap(
        df_lid,
        column="p",
        title=f"Target Recognition Probability by Trial Speaker ({experiment_dir.name})",
        xlabel="Probability",
        out_dir=out_dir,
        filename_base="probability_distribution_by_speaker_heatmap",
        baseline_val=1.0 / n_enrolments,
        baseline_label=f"Random Guess (p = 1/{n_enrolments})",
        speaker_metrics=speaker_metrics,
        summed_boundary_values=(1.0,),
        summed_boundary_epsilon=0.01,
        display_min=0.0,
        display_max=1.0,
        dpi=dpi,
    )


def plot_lid_distribution(experiment_dir, df_lid, out_dir):
    return plot_metric(
        df_lid,
        column="LID",
        title=f"Local Information Disclosure Distribution ({experiment_dir.name})",
        xlabel="Local Information Disclosure (bits)",
        out_dir=out_dir,
        filename_base="lid_distribution",
        baseline_val=0.0,
        baseline_label="No information disclosure",
        loc="upper left",
    )


def plot_lid_distribution_by_speaker(
    experiment_dir,
    df_lid,
    out_dir,
    speaker_metrics=None,
    dpi=300,
):
    n_enrolments = get_n_enrolments(experiment_dir, df_lid)

    return plot_metric_by_speaker(
        df_lid,
        column="LID",
        title=f"Local Information Disclosure by Trial Speaker ({experiment_dir.name})",
        xlabel="Local Information Disclosure (bits)",
        out_dir=out_dir,
        filename_base="lid_distribution_by_speaker",
        baseline_val=0.0,
        baseline_label="No information disclosure",
        loc="upper left",
        speaker_metrics=speaker_metrics,
        summed_boundary_values=(np.log2(n_enrolments),),
        summed_boundary_epsilon=-np.log2(0.99),
        display_max=np.log2(n_enrolments),
        dpi=dpi,
    )


def plot_lid_distribution_by_speaker_heatmap(
    experiment_dir,
    df_lid,
    out_dir,
    speaker_metrics=None,
    dpi=300,
):
    n_enrolments = get_n_enrolments(experiment_dir, df_lid)

    return plot_metric_by_speaker_heatmap(
        df_lid,
        column="LID",
        title=f"Local Information Disclosure by Trial Speaker ({experiment_dir.name})",
        xlabel="Local Information Disclosure (bits)",
        out_dir=out_dir,
        filename_base="lid_distribution_by_speaker_heatmap",
        baseline_val=0.0,
        baseline_label="No information disclosure",
        loc="upper left",
        speaker_metrics=speaker_metrics,
        summed_boundary_values=(np.log2(n_enrolments),),
        summed_boundary_epsilon=-np.log2(0.99),
        display_max=np.log2(n_enrolments),
        dpi=dpi,
    )


def regenerate_longitudinal_speaker_plots(
    experiment_dir,
    df_lid,
    speaker_metrics,
    dpi=300,
    out_dir=None,
):
    if out_dir is None:
        out_dir = RESULTS_DIR / experiment_dir.name / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context():
        configure_matplotlib()
        plot_paths = []
        plot_paths.extend(
            plot_probability_distribution_by_speaker(
                experiment_dir,
                df_lid,
                out_dir,
                speaker_metrics=speaker_metrics,
                dpi=dpi,
            )
        )
        plot_paths.extend(
            plot_probability_distribution_by_speaker_heatmap(
                experiment_dir,
                df_lid,
                out_dir,
                speaker_metrics=speaker_metrics,
                dpi=dpi,
            )
        )
        plot_paths.extend(
            plot_lid_distribution_by_speaker(
                experiment_dir,
                df_lid,
                out_dir,
                speaker_metrics=speaker_metrics,
                dpi=dpi,
            )
        )
        plot_paths.extend(
            plot_lid_distribution_by_speaker_heatmap(
                experiment_dir,
                df_lid,
                out_dir,
                speaker_metrics=speaker_metrics,
                dpi=dpi,
            )
        )

    return plot_paths


def experiment_plot_style(experiment_name):
    experiment_colors = {
        "B3": "tab:blue",
        "B4": "tab:orange",
        "B5": "tab:green",
        "T10-2": "tab:red",
        "T12-5": "tab:purple",
        "T25-1": "tab:brown",
        "T8-5": "tab:pink",
        "plain": "tab:gray",
        "random": "tab:olive",
    }
    experiment_styles = {
        "B3": "-",
        "B4": "--",
        "B5": "-.",
        "T10-2": ":",
        "T12-5": "-",
        "T25-1": "--",
        "T8-5": "-.",
        "plain": ":",
        "random": "-",
    }

    prefix = experiment_name.split("_", 1)[0]
    return (
        experiment_colors.get(prefix, "black"),
        experiment_styles.get(prefix, "-"),
    )


def plot_combined_lid_ccdf(all_lid_data, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.27, 4))
    ax.grid(True, which="both", ls="-", color="gray", alpha=0.2, zorder=0)

    for experiment_name, scores in sorted(all_lid_data.items()):
        valid_scores = scores[np.isfinite(scores)]
        if len(valid_scores) == 0:
            continue

        sorted_scores = np.sort(valid_scores)
        y_vals = np.arange(len(sorted_scores), 0, -1) / len(sorted_scores) * 100.0
        avg = float(np.mean(valid_scores))
        mx = float(np.max(valid_scores))
        pos_ratio = float(np.mean(valid_scores > 0))
        color, linestyle = experiment_plot_style(experiment_name)

        ax.plot(
            sorted_scores,
            y_vals,
            linewidth=1.5,
            label=f"{experiment_name} (Avg: {avg:.2f}, Max: {mx:.2f}, Pos: {pos_ratio:.1%})",
            color=color,
            linestyle=linestyle,
        )

    # ax.axvline(
    #     x=0.0,
    #     color="red",
    #     linestyle="dotted",
    #     linewidth=1.5,
    #     label="No information disclosure",
    # )
    ax.set_title("Combined Local Information Disclosure CCDF")
    ax.set_xlabel("Local Information Disclosure (bits)")
    ax.set_ylabel("Trials Exceeding Disclosure Level (%)")
    ax.set_ylim([0.0, 100.0])
    ax.legend(
        loc="lower left",
        framealpha=0.9,
        borderaxespad=0.2,
        borderpad=0.4,
        labelspacing=0.4,
    )

    plt.tight_layout()

    png_path = out_dir / "lid_combined_ccdf.png"
    pdf_path = out_dir / "lid_combined_ccdf.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return [png_path, pdf_path]


def process_experiment(experiment_dir):
    df_lid = load_lid_dataframe(experiment_dir)
    out_dir = RESULTS_DIR / experiment_dir.name / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = []
    plot_paths.extend(plot_probability_distribution(experiment_dir, df_lid, out_dir))
    plot_paths.extend(
        plot_probability_distribution_by_speaker(experiment_dir, df_lid, out_dir)
    )
    plot_paths.extend(
        plot_probability_distribution_by_speaker_heatmap(
            experiment_dir,
            df_lid,
            out_dir,
        )
    )
    plot_paths.extend(plot_lid_distribution(experiment_dir, df_lid, out_dir))
    plot_paths.extend(plot_lid_distribution_by_speaker(experiment_dir, df_lid, out_dir))
    plot_paths.extend(
        plot_lid_distribution_by_speaker_heatmap(experiment_dir, df_lid, out_dir)
    )

    return {
        "experiment": experiment_dir.name,
        "plots": len(plot_paths),
    }


def build_summary_table(experiment_dirs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for experiment_dir in experiment_dirs:
        metrics_path = RESULTS_DIR / experiment_dir.name / METRICS_FILE
        alt_metrics_path = (
            RESULTS_DIR / experiment_dir.name / ALTERNATIVE_METRICS_DIR_NAME / METRICS_FILE
        )

        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing required metrics file: {metrics_path}")

        metrics = json.loads(metrics_path.read_text())
        row = {
            "Experiment": experiment_dir.name,
            "ALID": float(metrics["ALID"]),
            "PDR": float(metrics["PDR"]),
            "NDR": float(metrics["NDR"]),
            "LID+": float(metrics["LID+"]) if metrics["LID+"] is not None else np.nan,
            "LID-": float(metrics["LID-"]) if metrics["LID-"] is not None else np.nan,
            "LID_max": float(metrics["LID_max"]),
        }

        if alt_metrics_path.is_file():
            alt_metrics = json.loads(alt_metrics_path.read_text())
            if "EER" in alt_metrics and alt_metrics["EER"] is not None:
                row["EER"] = float(alt_metrics["EER"])
            if "cllr" in alt_metrics and alt_metrics["cllr"] is not None:
                row["Cllr"] = float(alt_metrics["cllr"])

        rows.append(row)

    df_summary = pd.DataFrame(rows)
    optional_columns = ["EER", "Cllr"]
    empty_optional_columns = [
        column
        for column in optional_columns
        if column in df_summary and not df_summary[column].notna().any()
    ]
    df_summary = df_summary.drop(columns=empty_optional_columns)

    if "EER" in df_summary:
        df_summary = df_summary.sort_values("EER", na_position="last")
    else:
        df_summary = df_summary.sort_values("Experiment")
    df_summary = df_summary.reset_index(drop=True)

    ordered_columns = [
        column
        for column in [
            "Experiment",
            "EER",
            "Cllr",
            "ALID",
            "PDR",
            "NDR",
            "LID+",
            "LID-",
            "LID_max",
        ]
        if column in df_summary
    ]
    df_summary = df_summary[ordered_columns]

    summary_path = out_dir / "summary_table.csv"
    df_summary.to_csv(summary_path, index=False)
    return df_summary, summary_path


def build_calibration_coeff_table(experiment_dirs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for experiment_dir in experiment_dirs:
        experiment_calibration_path = calibration_parameters_path(experiment_dir)
        if not experiment_calibration_path.is_file():
            raise FileNotFoundError(
                f"Missing required calibration file: {experiment_calibration_path}"
            )

        calibration = json.loads(experiment_calibration_path.read_text())
        if "w" not in calibration or "b" not in calibration:
            raise ValueError(
                f"{experiment_calibration_path} is missing required keys: ['w', 'b']"
            )

        rows.append(
            {
                "experiment": experiment_dir.name,
                "w": float(calibration["w"]),
                "b": float(calibration["b"]),
            }
        )

    calibration_path = out_dir / "calibration_coeff.csv"
    pd.DataFrame(rows).to_csv(calibration_path, index=False)
    return calibration_path


def plot_summary_visuals(df_summary, out_dir):
    if "EER" not in df_summary:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    df_plot = df_summary[
        (df_summary["Experiment"] != "plain") & df_summary["EER"].notna()
    ].copy()
    if df_plot.empty:
        return []

    x_vals = df_plot["EER"]

    fig, ax = plt.subplots(figsize=(5.4, 3.1))

    experiment_colors = {
        "B3": "tab:blue",
        "B4": "tab:orange",
        "B5": "tab:green",
        "T10-2": "tab:red",
        "T12-5": "tab:purple",
        "T25-1": "tab:brown",
        "T8-5": "tab:pink",
        "plain": "tab:gray",
        "random": "tab:olive",
    }

    import matplotlib.lines as mlines

    for _, row in df_plot.iterrows():
        x_val = row["EER"]
        y_d_minus = row["LID-"]
        y_mi = row["ALID"]
        y_d_plus = row["LID+"]
        y_r_max = row["LID_max"]
        experiment = row["Experiment"]

        color = experiment_colors.get(experiment.split("_", 1)[0], "black")

        ax.scatter(x_val, y_d_minus, color=color, s=25, marker="v", zorder=3)
        ax.scatter(x_val, y_mi, color=color, s=25, marker="o", zorder=3)
        ax.scatter(x_val, y_d_plus, color=color, s=25, marker="^", zorder=3)
        ax.scatter(x_val, y_r_max, color=color, s=25, marker="x", zorder=3)

        y_values = [y for y in [y_d_minus, y_mi, y_d_plus, y_r_max] if np.isfinite(y)]
        if y_values:
            ax.plot(
                [x_val, x_val],
                [min(y_values), max(y_values)],
                color=color,
                linestyle="--",
                linewidth=1,
                zorder=2,
                alpha=0.6,
            )

    type_handles = [
        mlines.Line2D(
            [], [], color="gray", marker="x", linestyle="None", markersize=5, label="LID_max"
        ),
        mlines.Line2D(
            [], [], color="gray", marker="^", linestyle="None", markersize=5, label="LID+"
        ),
        mlines.Line2D(
            [], [], color="gray", marker="o", linestyle="None", markersize=5, label="ALID"
        ),
        mlines.Line2D(
            [], [], color="gray", marker="v", linestyle="None", markersize=5, label="LID-"
        ),
    ]

    metric_legend = ax.legend(
        handles=type_handles,
        fontsize=7,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        title="Metric",
        title_fontsize=7,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
    )
    ax.add_artist(metric_legend)

    experiment_handles = [
        mlines.Line2D(
            [],
            [],
            color=experiment_colors.get(experiment.split("_", 1)[0], "black"),
            marker="o",
            linestyle="None",
            markersize=5,
            label=experiment,
        )
        for experiment in df_plot["Experiment"]
    ]

    ax.legend(
        handles=experiment_handles,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="Experiment",
        title_fontsize=7,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        borderaxespad=0.0,
    )

    ax.axhline(0, color="gray", linestyle=":", linewidth=1, zorder=1, alpha=0.5)
    ax.set_xlabel("Equal Error Rate (EER)", fontsize=9)
    ax.set_ylabel("Information Disclosure (bits)", fontsize=9)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.tick_params(axis="both", which="major", labelsize=8)

    ax.set_xlim(max(0, x_vals.min() - 0.05), min(0.55, x_vals.max() + 0.05))

    all_y_vals = pd.concat(
        [
            df_plot["LID-"],
            df_plot["ALID"],
            df_plot["LID+"],
            df_plot["LID_max"],
        ]
    )
    all_y_vals = all_y_vals[np.isfinite(all_y_vals)]
    y_min = float(all_y_vals.min())
    y_max = float(all_y_vals.max())
    y_margin = (y_max - y_min) * 0.15
    if y_margin == 0:
        y_margin = 0.5
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    plt.tight_layout()

    scatter_png_path = out_dir / "eer_vs_infodisc_scatter.png"
    scatter_pdf_path = out_dir / "eer_vs_infodisc_scatter.pdf"
    plt.savefig(scatter_png_path, dpi=300, bbox_inches="tight")
    plt.savefig(scatter_pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return [scatter_png_path, scatter_pdf_path]


def format_table_value(value):
    if pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_dataframe_table(df):
    rows = [
        [format_table_value(row[column]) for column in df.columns]
        for _, row in df.iterrows()
    ]
    widths = [
        max(len(column), *(len(row[index]) for row in rows))
        for index, column in enumerate(df.columns)
    ]

    header_row = "  ".join(
        column.ljust(widths[index]) for index, column in enumerate(df.columns)
    )
    separator_row = "  ".join("-" * width for width in widths)
    print(header_row)
    print(separator_row)
    for row in rows:
        print(
            "  ".join(
                value.rjust(widths[index]) if index else value.ljust(widths[index])
                for index, value in enumerate(row)
            )
        )


def copy_png_figure(source_path, figure_name, out_dir, required=True):
    source = source_path.with_suffix(".png")
    if not source.is_file():
        if required:
            raise FileNotFoundError(f"Missing required figure source: {source}")
        return None

    destination = out_dir / f"{figure_name}.png"
    shutil.copy2(source, destination)
    return destination


def save_rounded_table(source_csv_path, table_name, out_dir):
    if not source_csv_path.is_file():
        raise FileNotFoundError(f"Missing required table source: {source_csv_path}")

    destination = out_dir / f"{table_name}.csv"
    df = pd.read_csv(source_csv_path)
    numeric_columns = df.select_dtypes(include="number").columns
    df[numeric_columns] = df[numeric_columns].mask(
        df[numeric_columns].abs() < 0.005,
        0.0,
    )
    df.round(2).to_csv(destination, index=False, float_format="%.2f")
    return destination


def export_paper_figures():
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    t10_plots_dir = RESULTS_DIR / "T10-2" / "plots"
    exported_paths = []
    skipped = []

    exported_paths.append(
        copy_png_figure(
            t10_plots_dir / "probability_distribution.pdf",
            "Figure2",
            PAPER_FIGURES_DIR,
        )
    )
    exported_paths.append(
        copy_png_figure(
            t10_plots_dir / "lid_distribution.png",
            "Figure3",
            PAPER_FIGURES_DIR,
        )
    )
    exported_paths.append(
        copy_png_figure(
            SUMMARY_DIR / "lid_combined_ccdf.png",
            "Figure4",
            PAPER_FIGURES_DIR,
        )
    )

    figure5_path = copy_png_figure(
        SUMMARY_DIR / "eer_vs_infodisc_scatter.png",
        "Figure5",
        PAPER_FIGURES_DIR,
        required=False,
    )
    if figure5_path:
        exported_paths.append(figure5_path)
    else:
        skipped.append("Figure5")

    exported_paths.append(
        save_rounded_table(
            SUMMARY_DIR / "calibration_coeff.csv",
            "Table3",
            PAPER_FIGURES_DIR,
        )
    )
    exported_paths.append(
        save_rounded_table(
            SUMMARY_DIR / "summary_table.csv",
            "Table4",
            PAPER_FIGURES_DIR,
        )
    )

    return {
        "files": len(exported_paths),
        "skipped": skipped,
    }


def plot_combined_summary(experiment_dirs):
    all_lid_data = {}
    for experiment_dir in experiment_dirs:
        df_lid = load_lid_dataframe(experiment_dir)
        all_lid_data[experiment_dir.name] = df_lid["LID"].to_numpy()

    summary_paths = []
    summary_paths.extend(plot_combined_lid_ccdf(all_lid_data, SUMMARY_DIR))
    df_summary, summary_table_path = build_summary_table(experiment_dirs, SUMMARY_DIR)
    summary_paths.append(summary_table_path)
    summary_paths.append(build_calibration_coeff_table(experiment_dirs, SUMMARY_DIR))
    eer_plot_paths = plot_summary_visuals(df_summary, SUMMARY_DIR)
    summary_paths.extend(eer_plot_paths)
    paper_export_summary = export_paper_figures()

    return {
        "summary_table": df_summary,
        "summary_files": len(summary_paths),
        "eer_plot_created": bool(eer_plot_paths),
        "paper_files": paper_export_summary["files"],
        "paper_skipped": paper_export_summary["skipped"],
    }


if __name__ == "__main__":
    configure_matplotlib()

    print()
    print(SEPARATOR)
    print("STEP 6. Plotting local information disclosure results.")
    print(SEPARATOR)
    print(
        "Creating probability/LID histograms, speaker ridgelines and heatmaps, "
        "combined summaries, and paper artifacts."
    )
    print()

    experiment_dirs = iter_experiment_dirs()
    experiment_summaries = []
    for experiment_dir in experiment_dirs:
        experiment_summaries.append(process_experiment(experiment_dir))
    summary = plot_combined_summary(experiment_dirs)

    total_experiment_plots = sum(item["plots"] for item in experiment_summaries)
    print(f"Generated {total_experiment_plots} per-experiment plot files.")
    print(f"Generated {summary['summary_files']} summary files.")
    print(f"Exported {summary['paper_files']} paper figure/table files.")
    if not summary["eer_plot_created"]:
        print("Skipped EER vs information disclosure plot because EER is unavailable.")
    for item in summary["paper_skipped"]:
        print(f"Skipped {item} paper export because its source figure is unavailable.")
    print()
    print_dataframe_table(summary["summary_table"])
    print()
    print("Plots are saved in results/experiments/<experiment>/plots/.")
    print("Summary outputs can be found in results/summary/.")
    print("Paper artifacts can be found in results/paper_figures/.")
    print(SEPARATOR2)
    print()
