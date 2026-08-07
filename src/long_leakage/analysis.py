"""Aggregate trial-level LLR vectors into longitudinal speaker evidence."""

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


REQUIRED_SCORE_COLUMNS = {"enroll_spk", "trial_spk", "trial_id", "llr"}
IDENTITY_COLUMNS = ["enroll_spk", "trial_spk", "trial_id"]
EXPERIMENT_COLORS = {
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
LID_METRIC_NAMES = ("ALID", "PDR", "NDR", "LID+", "LID-", "LID_max")
BEFORE_COLOR = "#287D8E"
AFTER_COLOR = "#E76F51"


def _stable_softmax(values):
    """Return row-wise log probabilities and probabilities."""
    row_max = np.max(values, axis=1, keepdims=True)
    shifted = values - row_max
    log_denominator = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    ln_p = shifted - log_denominator
    return ln_p, np.exp(ln_p)


def load_trial_scores(scores_csv_path):
    """Load and validate a complete trial-by-enrolment LLR matrix."""
    scores_csv_path = Path(scores_csv_path)
    df_scores = pd.read_csv(
        scores_csv_path,
        dtype={column: "string" for column in IDENTITY_COLUMNS},
    )
    missing_columns = REQUIRED_SCORE_COLUMNS - set(df_scores.columns)
    if missing_columns:
        raise ValueError(
            f"{scores_csv_path} is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    if df_scores.empty:
        raise ValueError(f"{scores_csv_path} contains no scores")

    null_identities = [
        column for column in IDENTITY_COLUMNS if df_scores[column].isna().any()
    ]
    if null_identities:
        raise ValueError(
            f"{scores_csv_path} contains missing identity values in: "
            f"{null_identities}"
        )

    df_scores["llr"] = pd.to_numeric(df_scores["llr"], errors="coerce")
    if not np.isfinite(df_scores["llr"].to_numpy()).all():
        raise ValueError(f"{scores_csv_path} contains non-finite LLR values")

    duplicate_rows = df_scores.duplicated(["trial_id", "enroll_spk"], keep=False)
    if duplicate_rows.any():
        examples = (
            df_scores.loc[duplicate_rows, ["trial_id", "enroll_spk"]]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Expected one score for every trial_id/enroll_spk pair. "
            f"Duplicate pairs: {examples}"
        )

    speakers_per_trial = df_scores.groupby("trial_id", sort=False)[
        "trial_spk"
    ].nunique()
    if (speakers_per_trial != 1).any():
        invalid_trials = speakers_per_trial[speakers_per_trial != 1].index[:10].tolist()
        raise ValueError(
            "Each trial_id must map to exactly one trial_spk. "
            f"Invalid trial_id values: {invalid_trials}"
        )

    n_enrolments = df_scores["enroll_spk"].nunique()
    enrolments_per_trial = df_scores.groupby("trial_id", sort=False)[
        "enroll_spk"
    ].nunique()
    if (enrolments_per_trial != n_enrolments).any():
        invalid_trials = enrolments_per_trial[
            enrolments_per_trial != n_enrolments
        ].index[:10].tolist()
        raise ValueError(
            "Every trial must contain the same enrolment set. "
            f"Invalid trial_id values: {invalid_trials}"
        )

    return df_scores


@dataclass(frozen=True)
class LongitudinalEvidence:
    """Matrices and identifiers used by the longitudinal analysis."""

    enroll_speakers: tuple[str, ...]
    trial_ids: tuple[str, ...]
    trial_speakers: tuple[str, ...]
    speaker_ids: tuple[str, ...]
    speaker_trial_counts: np.ndarray
    raw_llr: np.ndarray
    sum_llr: np.ndarray
    avg_llr: np.ndarray
    raw_ln_p: np.ndarray
    raw_p: np.ndarray
    sum_ln_p: np.ndarray
    sum_p: np.ndarray
    avg_ln_p: np.ndarray
    avg_p: np.ndarray

    @property
    def n_enrolments(self):
        return len(self.enroll_speakers)

    @property
    def n_trials(self):
        return len(self.trial_ids)

    @property
    def n_trial_speakers(self):
        return len(self.speaker_ids)

    @property
    def speaker_edges(self):
        return np.concatenate(([0], np.cumsum(self.speaker_trial_counts)))

    def scores_dataframe(self):
        rows = []
        for speaker_index, trial_spk in enumerate(self.speaker_ids):
            for enroll_index, enroll_spk in enumerate(self.enroll_speakers):
                rows.append(
                    {
                        "enroll_spk": enroll_spk,
                        "trial_spk": trial_spk,
                        "sum_llr": self.sum_llr[speaker_index, enroll_index],
                        "avg_llr": self.avg_llr[speaker_index, enroll_index],
                    }
                )
        return pd.DataFrame(
            rows,
            columns=["enroll_spk", "trial_spk", "sum_llr", "avg_llr"],
        )

    def probabilities_dataframe(self):
        rows = []
        for speaker_index, trial_spk in enumerate(self.speaker_ids):
            for enroll_index, enroll_spk in enumerate(self.enroll_speakers):
                rows.append(
                    {
                        "enroll_spk": enroll_spk,
                        "trial_spk": trial_spk,
                        "sum_p": self.sum_p[speaker_index, enroll_index],
                        "avg_p": self.avg_p[speaker_index, enroll_index],
                    }
                )
        return pd.DataFrame(
            rows,
            columns=["enroll_spk", "trial_spk", "sum_p", "avg_p"],
        )

    def mated_dataframe(self):
        enroll_index = {
            speaker_id: index
            for index, speaker_id in enumerate(self.enroll_speakers)
        }
        missing_speakers = sorted(set(self.speaker_ids) - set(enroll_index))
        if missing_speakers:
            raise ValueError(
                "No mated enrolment exists for trial_spk values: "
                f"{missing_speakers[:10]}"
            )

        rows = []
        log2_n = np.log2(self.n_enrolments)
        for trial_index, (trial_id, trial_spk) in enumerate(
            zip(self.trial_ids, self.trial_speakers)
        ):
            target_index = enroll_index[trial_spk]
            rows.append(
                {
                    "stage": "before",
                    "trial_id": trial_id,
                    "trial_spk": trial_spk,
                    "p": self.raw_p[trial_index, target_index],
                    "LID": log2_n
                    + self.raw_ln_p[trial_index, target_index] / np.log(2),
                }
            )

        for speaker_index, trial_spk in enumerate(self.speaker_ids):
            target_index = enroll_index[trial_spk]
            for stage, probabilities, log_probabilities in (
                ("summed", self.sum_p, self.sum_ln_p),
                ("averaged", self.avg_p, self.avg_ln_p),
            ):
                rows.append(
                    {
                        "stage": stage,
                        "trial_id": f"combined:{trial_spk}",
                        "trial_spk": trial_spk,
                        "p": probabilities[speaker_index, target_index],
                        "LID": log2_n
                        + log_probabilities[speaker_index, target_index]
                        / np.log(2),
                    }
                )

        return pd.DataFrame(
            rows,
            columns=["stage", "trial_id", "trial_spk", "p", "LID"],
        )


def build_longitudinal_evidence(df_scores):
    """Build raw and speaker-aggregated LLR and probability matrices."""
    input_enroll_speakers = df_scores["enroll_spk"].drop_duplicates().tolist()

    trial_metadata = df_scores[["trial_id", "trial_spk"]].drop_duplicates("trial_id")
    input_trial_speakers = trial_metadata["trial_spk"].drop_duplicates().tolist()
    trial_speaker_set = set(input_trial_speakers)
    enroll_speaker_set = set(input_enroll_speakers)
    shared_speakers = [
        speaker_id
        for speaker_id in input_enroll_speakers
        if speaker_id in trial_speaker_set
    ]
    enroll_speakers = tuple(
        shared_speakers
        + [
            speaker_id
            for speaker_id in input_enroll_speakers
            if speaker_id not in trial_speaker_set
        ]
    )
    speaker_ids = tuple(
        shared_speakers
        + [
            speaker_id
            for speaker_id in input_trial_speakers
            if speaker_id not in enroll_speaker_set
        ]
    )

    ordered_metadata_parts = [
        trial_metadata[trial_metadata["trial_spk"] == speaker_id]
        for speaker_id in speaker_ids
    ]
    ordered_metadata = pd.concat(ordered_metadata_parts, ignore_index=True)
    trial_ids = tuple(ordered_metadata["trial_id"].tolist())
    trial_speakers = tuple(ordered_metadata["trial_spk"].tolist())

    raw_llr = (
        df_scores.pivot(index="trial_id", columns="enroll_spk", values="llr")
        .reindex(index=trial_ids, columns=enroll_speakers)
        .to_numpy(dtype=float)
    )
    if not np.isfinite(raw_llr).all():
        raise ValueError("The trial-by-enrolment LLR matrix is incomplete")

    speaker_trial_counts = np.array(
        [trial_speakers.count(speaker_id) for speaker_id in speaker_ids],
        dtype=int,
    )
    split_points = np.cumsum(speaker_trial_counts)[:-1]
    sum_llr = np.stack(
        [matrix.sum(axis=0) for matrix in np.split(raw_llr, split_points)]
    )
    avg_llr = sum_llr / speaker_trial_counts[:, np.newaxis]

    raw_ln_p, raw_p = _stable_softmax(raw_llr)
    sum_ln_p, sum_p = _stable_softmax(sum_llr)
    avg_ln_p, avg_p = _stable_softmax(avg_llr)

    return LongitudinalEvidence(
        enroll_speakers=enroll_speakers,
        trial_ids=trial_ids,
        trial_speakers=trial_speakers,
        speaker_ids=speaker_ids,
        speaker_trial_counts=speaker_trial_counts,
        raw_llr=raw_llr,
        sum_llr=sum_llr,
        avg_llr=avg_llr,
        raw_ln_p=raw_ln_p,
        raw_p=raw_p,
        sum_ln_p=sum_ln_p,
        sum_p=sum_p,
        avg_ln_p=avg_ln_p,
        avg_p=avg_p,
    )


def _selected_tick_indices(length, maximum):
    if length <= maximum:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, maximum, dtype=int))


def _set_x_ticks(ax, labels):
    indices = _selected_tick_indices(len(labels), 40)
    ax.set_xticks(indices + 0.5, [labels[index] for index in indices])
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.set_xlabel("Enrolment speaker")


def _set_raw_y_ticks(ax, evidence):
    indices = _selected_tick_indices(evidence.n_trials, 16)
    ax.set_yticks(indices + 0.5, [evidence.trial_ids[index] for index in indices])
    ax.tick_params(axis="y", labelsize=6)
    ax.set_ylabel("Trial")


def _set_aggregate_y_ticks(ax, evidence):
    edges = evidence.speaker_edges
    indices = _selected_tick_indices(evidence.n_trial_speakers, 40)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.set_yticks(
        centers[indices],
        [evidence.speaker_ids[index] for index in indices],
    )
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Trial speaker")


def _llr_norm(values):
    maximum = float(np.max(np.abs(values)))
    if maximum == 0:
        maximum = 1.0
    return TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)


def _mated_marker_coordinates(evidence):
    enroll_indices = {
        speaker_id: index
        for index, speaker_id in enumerate(evidence.enroll_speakers)
    }
    speaker_edges = evidence.speaker_edges
    coordinates = [
        (
            enroll_indices[trial_spk] + 0.5,
            (speaker_edges[index] + speaker_edges[index + 1]) / 2,
        )
        for index, trial_spk in enumerate(evidence.speaker_ids)
        if trial_spk in enroll_indices
    ]
    if not coordinates:
        return np.empty(0), np.empty(0)
    x_values, y_values = zip(*coordinates)
    return np.asarray(x_values), np.asarray(y_values)


def _plot_mated_markers(ax, evidence):
    x_values, y_values = _mated_marker_coordinates(evidence)
    ax.scatter(
        x_values,
        y_values,
        s=7,
        marker="o",
        facecolors="white",
        edgecolors="black",
        linewidths=0.25,
        zorder=3,
    )


def _save_figure(fig, output_dir, filename_base, dpi):
    paths = []
    for suffix in ("png", "pdf"):
        output_path = output_dir / f"{filename_base}.{suffix}"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        paths.append(output_path)
    plt.close(fig)
    return paths


def plot_evidence_heatmaps(evidence, output_dir, dpi=300):
    """Plot raw, summed, and averaged LLR matrices at equal physical heights."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 8),
        constrained_layout=True,
    )
    x_edges = np.arange(evidence.n_enrolments + 1)
    raw_y_edges = np.arange(evidence.n_trials + 1)
    aggregate_y_edges = evidence.speaker_edges
    panels = (
        (
            evidence.raw_llr,
            raw_y_edges,
            "Individual trial LLRs",
            True,
            "RdBu_r",
        ),
        (
            evidence.sum_llr,
            aggregate_y_edges,
            "Summed LLRs",
            False,
            "RdBu_r",
        ),
        (
            evidence.avg_llr,
            aggregate_y_edges,
            "Average LLRs",
            False,
            "RdBu_r",
        ),
    )

    for ax, (values, y_edges, title, is_raw, cmap) in zip(axes, panels):
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            values,
            cmap=cmap,
            norm=_llr_norm(values),
            shading="flat",
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlim(0, evidence.n_enrolments)
        ax.set_ylim(evidence.n_trials, 0)
        _set_x_ticks(ax, evidence.enroll_speakers)
        if is_raw:
            _set_raw_y_ticks(ax, evidence)
        else:
            _set_aggregate_y_ticks(ax, evidence)
            _plot_mated_markers(ax, evidence)
        fig.colorbar(mesh, ax=ax, label="LLR", shrink=0.85)

    return _save_figure(fig, output_dir, "llr_heatmaps", dpi)


def plot_probability_heatmaps(evidence, output_dir, dpi=300):
    """Plot normalized probabilities using the same vertical row geometry."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 8),
        constrained_layout=True,
    )
    x_edges = np.arange(evidence.n_enrolments + 1)
    raw_y_edges = np.arange(evidence.n_trials + 1)
    aggregate_y_edges = evidence.speaker_edges
    panels = (
        (
            evidence.raw_p,
            raw_y_edges,
            "Individual trial probabilities",
            True,
            "viridis",
        ),
        (
            evidence.sum_p,
            aggregate_y_edges,
            "Summed-evidence probabilities",
            False,
            "viridis",
        ),
        (
            evidence.avg_p,
            aggregate_y_edges,
            "Average-evidence probabilities",
            False,
            "viridis",
        ),
    )

    for ax, (values, y_edges, title, is_raw, cmap) in zip(axes, panels):
        maximum = float(values.max())
        if maximum == 0:
            maximum = 1.0
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            values,
            cmap=cmap,
            norm=Normalize(vmin=0.0, vmax=maximum),
            shading="flat",
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlim(0, evidence.n_enrolments)
        ax.set_ylim(evidence.n_trials, 0)
        _set_x_ticks(ax, evidence.enroll_speakers)
        if is_raw:
            _set_raw_y_ticks(ax, evidence)
        else:
            _set_aggregate_y_ticks(ax, evidence)
            _plot_mated_markers(ax, evidence)
        fig.colorbar(mesh, ax=ax, label="Probability", shrink=0.85)

    return _save_figure(fig, output_dir, "probability_heatmaps", dpi)


def _relative_histogram(ax, values, bins, label, color):
    ax.hist(
        values,
        bins=bins,
        weights=np.full(len(values), 1.0 / len(values)),
        histtype="bar",
        linewidth=0.7,
        color=color,
        edgecolor=color,
        alpha=0.45,
        label=f"{label} (n={len(values)})",
    )


def plot_mated_comparisons(
    evidence,
    mated,
    output_dir,
    dpi=300,
    after_stage="summed",
    after_label="After summing",
):
    """Plot target probability and LID before and after aggregation."""
    before = mated[mated["stage"] == "before"]
    after = mated[mated["stage"] == after_stage]
    if before.empty or after.empty:
        raise ValueError(
            f"Mated data must contain before and {after_stage} stages"
        )

    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    probability_bins = np.linspace(0.0, 1.0, 41)
    _relative_histogram(
        ax, before["p"].to_numpy(), probability_bins, "Before", "#6B7280"
    )
    _relative_histogram(
        ax,
        after["p"].to_numpy(),
        probability_bins,
        after_label,
        AFTER_COLOR,
    )
    ax.axvline(
        1.0 / evidence.n_enrolments,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"Random guess (1/{evidence.n_enrolments})",
    )
    ax.set_xlabel("Mated enrolment probability")
    ax.set_ylabel("Relative frequency")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.legend()
    probability_paths = _save_figure(
        fig, output_dir, "mated_probability_comparison", dpi
    )

    before_lid = before["LID"].to_numpy()
    after_lid = after["LID"].to_numpy()
    combined_lid = np.concatenate((before_lid, after_lid))
    if np.ptp(combined_lid) == 0:
        lid_bins = np.linspace(combined_lid[0] - 0.5, combined_lid[0] + 0.5, 41)
    else:
        lid_bins = np.linspace(combined_lid.min(), combined_lid.max(), 41)

    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    _relative_histogram(ax, before_lid, lid_bins, "Before", "#6B7280")
    _relative_histogram(ax, after_lid, lid_bins, after_label, AFTER_COLOR)
    ax.axvline(
        0.0,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="No information disclosure",
    )
    ax.set_xlabel("Local Information Disclosure (bits)")
    ax.set_ylabel("Relative frequency")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    lid_paths = _save_figure(fig, output_dir, "mated_lid_comparison", dpi)

    return probability_paths + lid_paths


def aggregate_lid_metrics(
    experiment_name,
    lid_values,
    n_enrolments,
    aggregation="averaged_llr",
):
    """Compute the existing pipeline's LID summary metrics."""
    lid_values = np.asarray(lid_values, dtype=float)
    if len(lid_values) == 0:
        raise ValueError(f"No LID values found for experiment: {experiment_name}")
    if not np.isfinite(lid_values).all():
        raise ValueError(
            f"Non-finite LID values found for experiment: {experiment_name}"
        )

    positive = lid_values[lid_values > 0]
    negative = lid_values[lid_values <= 0]
    return {
        "experiment": experiment_name,
        "aggregation": aggregation,
        "n_trials": int(len(lid_values)),
        "n_enrolments": int(n_enrolments),
        "ALID": float(lid_values.mean()),
        "PDR": float(len(positive) / len(lid_values)),
        "NDR": float(len(negative) / len(lid_values)),
        "LID+": float(positive.mean()) if len(positive) else None,
        "LID-": float(negative.mean()) if len(negative) else None,
        "LID_max": float(lid_values.max()),
    }


def _metric_snapshot(metrics):
    return {
        "n_trials": metrics["n_trials"],
        **{name: metrics[name] for name in LID_METRIC_NAMES},
    }


def _metric_changes(before, after):
    changes = {}
    for name in LID_METRIC_NAMES:
        before_value = before[name]
        after_value = after[name]
        changes[name] = (
            float(after_value - before_value)
            if before_value is not None and after_value is not None
            else None
        )
    return changes


def _format_bar_value(value):
    rounded = round(float(value), 2)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.2f}"


def plot_lid_metrics_before_after(
    experiment_name,
    before_metrics,
    after_metrics,
    output_dir,
    dpi=300,
):
    """Plot the six LID summary metrics before and after aggregation."""
    x_positions = np.arange(len(LID_METRIC_NAMES))
    width = 0.36
    rate_metrics = {"PDR", "NDR"}

    fig, percent_ax = plt.subplots(figsize=(7.8, 4.3), constrained_layout=True)
    bits_ax = percent_ax.twinx()
    bits_ax.patch.set_visible(False)

    legend_handles = {}
    for stage, metrics, offset, color in (
        ("Before aggregation", before_metrics, -width / 2, BEFORE_COLOR),
        ("After aggregation", after_metrics, width / 2, AFTER_COLOR),
    ):
        for index, metric_name in enumerate(LID_METRIC_NAMES):
            value = metrics[metric_name]
            is_rate = metric_name in rate_metrics
            axis = percent_ax if is_rate else bits_ax
            plotted_value = (
                0.0
                if value is None
                else float(value) * 100.0
                if is_rate
                else float(value)
            )
            bar = axis.bar(
                index + offset,
                plotted_value,
                width,
                color=color,
                zorder=2,
            )[0]
            legend_handles.setdefault(stage, bar)

            if value is None:
                label = "n/a"
            elif is_rate:
                label = f"{float(value) * 100.0:.2f}%"
            else:
                label = _format_bar_value(value)
            axis.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                clip_on=False,
                zorder=4,
            )

    percent_values = [
        float(metrics[name]) * 100.0
        for metrics in (before_metrics, after_metrics)
        for name in rate_metrics
        if metrics[name] is not None
    ]
    percent_top = max(100.0, *percent_values) * 1.14
    percent_ax.set_ylim(0.0, percent_top)

    bit_values = [
        float(metrics[name])
        for metrics in (before_metrics, after_metrics)
        for name in LID_METRIC_NAMES
        if name not in rate_metrics and metrics[name] is not None
    ]
    bit_min = min(0.0, *bit_values)
    bit_max = max(0.0, *bit_values)
    bit_span = bit_max - bit_min
    if bit_span == 0:
        bit_span = 1.0
    bits_ax.set_ylim(bit_min - bit_span * 0.16, bit_max + bit_span * 0.18)

    bits_ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
    percent_ax.set_title(
        f"LID Metrics Before and After Aggregation ({experiment_name})"
    )
    percent_ax.set_ylabel("Disclosure rate (%)")
    bits_ax.set_ylabel("Information disclosure (bits)")
    percent_ax.set_xticks(x_positions, LID_METRIC_NAMES)
    percent_ax.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        framealpha=0.9,
    )
    percent_ax.grid(axis="y", color="gray", alpha=0.2, zorder=0)
    return _save_figure(
        fig,
        output_dir,
        "lid_metrics_before_after",
        dpi,
    )


def _experiment_color(experiment_name, index):
    prefix = experiment_name.split("_", 1)[0]
    if prefix in EXPERIMENT_COLORS:
        return EXPERIMENT_COLORS[prefix]
    return plt.get_cmap("tab20")(index % 20)


def _plot_ccdf_curve(ax, values, color, linestyle, label=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    sorted_values = np.sort(values)

    plot_values = np.concatenate((sorted_values, sorted_values[-1:]))
    percentages = (
        np.arange(len(values), -1, -1, dtype=float) / len(values) * 100.0
    )
    ax.plot(
        plot_values,
>>>>>>> Stashed changes
        percentages,
        color=color,
        linestyle=linestyle,
        linewidth=1.5,
        label=label,
    )


def plot_averaged_lid_ccdf(averaged_lids, output_dir, dpi=300):
    """Plot one CCDF per experiment for LID from averaged LLR vectors."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    ax.grid(True, color="gray", alpha=0.2, zorder=0)

    for index, (experiment_name, values) in enumerate(sorted(averaged_lids.items())):
        values = np.asarray(values, dtype=float)
        label = (
            f"{experiment_name} "
            f"(Avg: {values.mean():.2f}, Max: {values.max():.2f}, "
            f"Pos: {np.mean(values > 0):.1%})"
        )
        _plot_ccdf_curve(
            ax,
            values,
            color=_experiment_color(experiment_name, index),
            linestyle="-",
            label=label,
        )

    ax.set_title("Longitudinal LID CCDF (Averaged LLR Vectors)")
    ax.set_xlabel("Local Information Disclosure (bits)")
    ax.set_ylabel("Speaker observations exceeding disclosure level (%)")
    ax.set_ylim(0.0, 100.0)
    ax.legend(
        loc="lower left",
        fontsize=7,
        framealpha=0.9,
        borderaxespad=0.2,
        borderpad=0.4,
        labelspacing=0.4,
    )
    return _save_figure(fig, output_dir, "lid_combined_ccdf", dpi)


def plot_original_vs_averaged_lid_ccdf(
    original_lids,
    averaged_lids,
    output_dir,
    dpi=300,
):
    """Compare original and averaged LID using color for experiment and line style."""
    experiment_names = sorted(set(original_lids) & set(averaged_lids))
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    ax.grid(True, color="gray", alpha=0.2, zorder=0)

    colors = {}
    for index, experiment_name in enumerate(experiment_names):
        color = _experiment_color(experiment_name, index)
        colors[experiment_name] = color
        _plot_ccdf_curve(
            ax,
            original_lids[experiment_name],
            color=color,
            linestyle=":",
        )
        _plot_ccdf_curve(
            ax,
            averaged_lids[experiment_name],
            color=color,
          linestyle="-",

        )

    experiment_handles = [
        Line2D([0], [0], color=colors[name], linewidth=1.8, label=name)
        for name in experiment_names
    ]
    experiment_legend = ax.legend(
        handles=experiment_handles,
        title="Experiment",
        loc="lower left",
        fontsize=7,
        title_fontsize=7,
        ncol=2,
        framealpha=0.9,
    )
    ax.add_artist(experiment_legend)
    stage_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.8,
            linestyle="-",
            label="Original trial LID",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.8,
            linestyle=":",
            label="Averaged LLR LID",
        ),
    ]
    ax.legend(
        handles=stage_handles,
        title="Evidence",
        loc="upper right",
        fontsize=7,
        title_fontsize=7,
        framealpha=0.9,
    )

    ax.set_title("Original vs Longitudinal Local Information Disclosure")
    ax.set_xlabel("Local Information Disclosure (bits)")
    ax.set_ylabel("Observations exceeding disclosure level (%)")
    ax.set_ylim(0.0, 100.0)
    return _save_figure(
        fig,
        output_dir,
        "lid_original_vs_averaged_ccdf",
        dpi,
    )


def plot_metric_experiment_summary(metrics_by_experiment, output_dir, dpi=300):
    """Plot before/after values for every metric and experiment."""
    experiment_names = [
        metrics["experiment"] for metrics in metrics_by_experiment
    ]
    x_positions = np.arange(len(experiment_names))
    width = 0.34
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12.5, 11),
        constrained_layout=True,
    )
    axes = axes.ravel()
    panel_metrics = ("PDR", "NDR", "ALID", "LID+", "LID-", "LID_max")
    rate_metrics = {"PDR", "NDR"}

    for ax, metric_name in zip(axes, panel_metrics):
        all_values = []
        for stage_key, offset, color in (
            ("before_aggregation", -width / 2, BEFORE_COLOR),
            ("after_aggregation", width / 2, AFTER_COLOR),
        ):
            values = [
                metrics[stage_key][metric_name]
                for metrics in metrics_by_experiment
            ]
            if metric_name in rate_metrics:
                values = [
                    float(value) * 100.0 if value is not None else None
                    for value in values
                ]
            finite_values = [
                float(value) if value is not None else 0.0 for value in values
            ]
            all_values.extend(value for value in values if value is not None)
            ax.bar(
                x_positions + offset,
                finite_values,
                width,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                zorder=2,
            )
            for index, value in enumerate(values):
                if value is None:
                    ax.annotate(
                        "n/a",
                        xy=(index + offset, 0.0),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=5.5,
                    )

        ax.set_title(metric_name)
        if metric_name in rate_metrics:
            ax.set_ylabel("Disclosure rate (%)")
            ax.set_ylim(0.0, 105.0)
        else:
            ax.axhline(0.0, color="black", linewidth=0.7, zorder=1)
            ax.set_ylabel("Bits")
            if all_values:
                value_min = min(0.0, *all_values)
                value_max = max(0.0, *all_values)
                value_span = value_max - value_min
                if value_span == 0:
                    value_span = 1.0
                ax.set_ylim(
                    value_min - value_span * 0.12,
                    value_max + value_span * 0.12,
                )
            else:
                ax.set_ylim(-0.5, 0.5)

    for ax in axes:
        ax.set_xticks(x_positions, experiment_names)
        ax.tick_params(axis="x", labelrotation=55, labelsize=7)
        ax.grid(axis="y", color="gray", alpha=0.18, zorder=0)

    stage_handles = [
        Patch(facecolor=BEFORE_COLOR, label="Before aggregation"),
        Patch(facecolor=AFTER_COLOR, label="After aggregation"),
    ]
    fig.legend(
        handles=stage_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        framealpha=0.9,
    )
    fig.suptitle("Longitudinal LID Metrics Across Experiments", fontsize=14)
    return _save_figure(
        fig,
        output_dir,
        "lid_metrics_experiment_comparison",
        dpi,
    )


def create_longitudinal_summary(experiment_dirs, output_dir, dpi=300):
    """Create cross-experiment metrics and CCDF plots for averaged LLR evidence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_experiment = []
    original_lids = {}
    averaged_lids = {}
    plot_paths = []
    for experiment_dir in experiment_dirs:
        experiment_dir = Path(experiment_dir)
        experiment_name = experiment_dir.name
        mated_path = experiment_dir / "long" / "mated_probabilities.csv"
        summary_path = experiment_dir / "long" / "summary.json"
        if not mated_path.is_file():
            raise FileNotFoundError(f"Missing longitudinal mated file: {mated_path}")
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing longitudinal summary file: {summary_path}")

        mated = pd.read_csv(mated_path)
        required_columns = {"stage", "LID"}
        missing_columns = required_columns - set(mated.columns)
        if missing_columns:
            raise ValueError(
                f"{mated_path} is missing required columns: {sorted(missing_columns)}"
            )

        original = mated.loc[mated["stage"] == "before", "LID"].to_numpy()
        averaged = mated.loc[mated["stage"] == "averaged", "LID"].to_numpy()
        if len(original) == 0 or len(averaged) == 0:
            raise ValueError(
                f"{mated_path} must contain both before and averaged LID values"
            )

        experiment_summary = json.loads(summary_path.read_text())
        before_metrics = aggregate_lid_metrics(
            experiment_name,
            original,
            experiment_summary["n_enrolments"],
            aggregation="individual_trials",
        )
        metrics = aggregate_lid_metrics(
            experiment_name,
            averaged,
            experiment_summary["n_enrolments"],
        )
        metrics["before_aggregation"] = _metric_snapshot(before_metrics)
        metrics["after_aggregation"] = _metric_snapshot(metrics)
        metrics["change_after_minus_before"] = _metric_changes(
            before_metrics,
            metrics,
        )
        experiment_output_dir = output_dir / experiment_name
        experiment_output_dir.mkdir(parents=True, exist_ok=True)
        results_path = experiment_output_dir / "results.json"
        results_path.write_text(json.dumps(metrics, indent=2) + "\n")
        plot_paths.extend(
            plot_lid_metrics_before_after(
                experiment_name,
                metrics["before_aggregation"],
                metrics["after_aggregation"],
                experiment_output_dir,
                dpi=dpi,
            )
        )

        metrics_by_experiment.append(metrics)
        original_lids[experiment_name] = original
        averaged_lids[experiment_name] = averaged

    table_rows = [
        {
            "Experiment": metrics["experiment"],
            "ALID": metrics["ALID"],
            "PDR": metrics["PDR"],
            "NDR": metrics["NDR"],
            "LID+": metrics["LID+"],
            "LID-": metrics["LID-"],
            "LID_max": metrics["LID_max"],
        }
        for metrics in metrics_by_experiment
    ]
    summary_table = pd.DataFrame(table_rows).sort_values("Experiment")
    summary_table_path = output_dir / "summary_table.csv"
    summary_table.to_csv(summary_table_path, index=False)

    plot_paths.extend(plot_averaged_lid_ccdf(averaged_lids, output_dir, dpi=dpi))
    plot_paths.extend(
        plot_original_vs_averaged_lid_ccdf(
            original_lids,
            averaged_lids,
            output_dir,
            dpi=dpi,
        )
    )
    plot_paths.extend(
        plot_metric_experiment_summary(
            metrics_by_experiment,
            output_dir,
            dpi=dpi,
        )
    )
    return {
        "experiments": len(metrics_by_experiment),
        "summary_table_path": summary_table_path,
        "plot_paths": plot_paths,
    }


def _aggregation_scores_dataframe(
    evidence,
    llr_values,
    ln_p_values,
    probability_values,
):
    rows = []
    for speaker_index, trial_spk in enumerate(evidence.speaker_ids):
        for enroll_index, enroll_spk in enumerate(evidence.enroll_speakers):
            rows.append(
                {
                    "enroll_spk": enroll_spk,
                    "trial_spk": trial_spk,
                    "n_trials": int(
                        evidence.speaker_trial_counts[speaker_index]
                    ),
                    "llr": llr_values[speaker_index, enroll_index],
                    "ln_p": ln_p_values[speaker_index, enroll_index],
                    "p": probability_values[speaker_index, enroll_index],
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "enroll_spk",
            "trial_spk",
            "n_trials",
            "llr",
            "ln_p",
            "p",
        ],
    )


def _write_aggregation_method_outputs(
    evidence,
    mated,
    output_dir,
    method_name,
    source_stage,
    llr_values,
    ln_p_values,
    probability_values,
    after_label,
    dpi,
):
    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    method_scores = _aggregation_scores_dataframe(
        evidence,
        llr_values,
        ln_p_values,
        probability_values,
    )
    method_scores.to_csv(method_dir / "scores.csv", index=False)

    method_mated = mated[mated["stage"].isin(("before", source_stage))].copy()
    method_mated["stage"] = method_mated["stage"].replace(source_stage, "after")
    method_mated.to_csv(method_dir / "mated_probabilities.csv", index=False)

    method_summary = {
        "aggregation": method_name,
        "n_enrolments": evidence.n_enrolments,
        "n_input_trials": evidence.n_trials,
        "n_output_speakers": evidence.n_trial_speakers,
        "min_trials_per_speaker": int(evidence.speaker_trial_counts.min()),
        "max_trials_per_speaker": int(evidence.speaker_trial_counts.max()),
    }
    (method_dir / "summary.json").write_text(
        json.dumps(method_summary, indent=2) + "\n"
    )

    return plot_mated_comparisons(
        evidence,
        method_mated,
        method_dir,
        dpi=dpi,
        after_stage="after",
        after_label=after_label,
    )


def analyse_experiment(experiment_dir, dpi=300):
    """Create longitudinal data and plots for one experiment directory."""
    experiment_dir = Path(experiment_dir)
    test_scores_path = experiment_dir / "scores" / "test_scores.csv"
    if not test_scores_path.is_file():
        raise FileNotFoundError(f"Missing required test scores file: {test_scores_path}")

    df_scores = load_trial_scores(test_scores_path)
    evidence = build_longitudinal_evidence(df_scores)
    mated = evidence.mated_dataframe()

    output_dir = experiment_dir / "long"
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = output_dir / "scores.csv"
    probabilities_path = output_dir / "probabilities.csv"
    mated_path = output_dir / "mated_probabilities.csv"
    summary_path = output_dir / "summary.json"

    evidence.scores_dataframe().to_csv(scores_path, index=False)
    evidence.probabilities_dataframe().to_csv(probabilities_path, index=False)
    mated.to_csv(mated_path, index=False)

    summary = {
        "experiment": experiment_dir.name,
        "n_enrolments": evidence.n_enrolments,
        "n_trials": evidence.n_trials,
        "n_trial_speakers": evidence.n_trial_speakers,
        "min_trials_per_speaker": int(evidence.speaker_trial_counts.min()),
        "max_trials_per_speaker": int(evidence.speaker_trial_counts.max()),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    plot_paths = []
    plot_paths.extend(plot_evidence_heatmaps(evidence, output_dir, dpi=dpi))
    plot_paths.extend(plot_probability_heatmaps(evidence, output_dir, dpi=dpi))
    plot_paths.extend(plot_mated_comparisons(evidence, mated, output_dir, dpi=dpi))
    plot_paths.extend(
        _write_aggregation_method_outputs(
            evidence,
            mated,
            output_dir,
            method_name="sum_llr",
            source_stage="summed",
            llr_values=evidence.sum_llr,
            ln_p_values=evidence.sum_ln_p,
            probability_values=evidence.sum_p,
            after_label="After summing LLRs",
            dpi=dpi,
        )
    )
    plot_paths.extend(
        _write_aggregation_method_outputs(
            evidence,
            mated,
            output_dir,
            method_name="average_llr",
            source_stage="averaged",
            llr_values=evidence.avg_llr,
            ln_p_values=evidence.avg_ln_p,
            probability_values=evidence.avg_p,
            after_label="After averaging LLRs",
            dpi=dpi,
        )
    )

    return {
        **summary,
        "scores_path": scores_path,
        "probabilities_path": probabilities_path,
        "mated_path": mated_path,
        "summary_path": summary_path,
        "plot_paths": plot_paths,
    }
