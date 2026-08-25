"""Explore how LID changes as more trials from one speaker are combined.

For every evidence count ``k``, the analysis considers the population of all
unique, unordered, within-speaker subsets of ``k`` trials.  Order is irrelevant
because the LLR vectors are summed.  Small populations are enumerated exactly;
larger populations are represented by a reproducible uniform sample.

Four aggregation rules are evaluated on exactly the same subsets:

* summed: softmax(sum(LLR)), equivalent to T = 1
* averaged: softmax(sum(LLR) / k), equivalent to T = k
* NLL-calibrated: softmax(sum(LLR) / T_NLL)
* Brier-calibrated: softmax(sum(LLR) / T_Brier)

``T_NLL`` and ``T_Brier`` are fixed across ``k`` and read from the existing
development-only longitudinal calibration output for each experiment.
"""

import argparse
from bisect import bisect_right
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations, islice
import json
from math import comb, log2, sqrt
from pathlib import Path
import random
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from long_leakage.analysis import (
    LongitudinalEvidence,
    build_longitudinal_evidence,
    load_trial_scores,
)
from tools.utils import (
    PROJECT_ROOT,
    SEPARATOR,
    SEPARATOR2,
    iter_experiment_dirs,
    relative_path,
)


METHODS = (
    "sum_llr",
    "average_llr",
    "temperature_scaled_nll",
    "temperature_scaled_brier",
)
METHOD_STYLES = {
    "sum_llr": {
        "label": "Summed evidence",
        "formula": "T = 1",
        "color": "#D97706",
    },
    "average_llr": {
        "label": "Averaged evidence",
        "formula": "T = k",
        "color": "#2563EB",
    },
    "temperature_scaled_nll": {
        "label": "NLL-fit temperature",
        "formula": "T = {nll:.6g}",
        "color": "#0F766E",
    },
    "temperature_scaled_brier": {
        "label": "Brier-fit temperature",
        "formula": "T = {brier:.6g}",
        "color": "#C026D3",
    },
}
DEFAULT_MAX_COMBINATIONS = 50_000


@dataclass(frozen=True)
class CombinationPopulation:
    """LID draws and population metadata for one evidence count."""

    k: int
    total_combinations: int
    eligible_speakers: int
    evaluated_combinations: int
    is_exact: bool
    values: dict[str, np.ndarray]


def _derived_seed(seed, k):
    digest = sha256(f"{seed}:{k}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _sample_unique_ranks(population_size, sample_size, rng):
    """Use Floyd's algorithm to sample arbitrary-size integer ranges."""
    if sample_size > population_size:
        raise ValueError("sample_size cannot exceed population_size")
    selected = set()
    start = population_size - sample_size
    for offset in range(sample_size):
        upper = start + offset
        candidate = rng.randrange(upper + 1)
        selected.add(upper if candidate in selected else candidate)
    return sorted(selected)


def _unrank_combination(n_items, n_chosen, rank):
    """Return the lexicographic combination at a zero-based integer rank."""
    total = comb(n_items, n_chosen)
    if rank < 0 or rank >= total:
        raise ValueError(f"Combination rank {rank} is outside [0, {total})")

    result = []
    next_item = 0
    remaining = n_chosen
    while remaining:
        last_candidate = n_items - remaining
        for candidate in range(next_item, last_candidate + 1):
            block_size = comb(n_items - candidate - 1, remaining - 1)
            if rank < block_size:
                result.append(candidate)
                next_item = candidate + 1
                remaining -= 1
                break
            rank -= block_size
    return tuple(result)


def _speaker_trial_matrices(evidence):
    enroll_index = {
        speaker_id: index
        for index, speaker_id in enumerate(evidence.enroll_speakers)
    }
    missing = [
        speaker_id
        for speaker_id in evidence.speaker_ids
        if speaker_id not in enroll_index
    ]
    if missing:
        raise ValueError(
            "Combination convergence requires a mated enrolment for every "
            f"trial speaker. Missing: {missing[:10]}"
        )

    matrices = []
    targets = []
    for speaker_index, speaker_id in enumerate(evidence.speaker_ids):
        start = int(evidence.speaker_edges[speaker_index])
        stop = int(evidence.speaker_edges[speaker_index + 1])
        matrices.append(evidence.raw_llr[start:stop])
        targets.append(enroll_index[speaker_id])
    return matrices, np.asarray(targets, dtype=int)


def _combination_indices_by_speaker(counts, k, max_combinations, seed):
    eligible = [index for index, count in enumerate(counts) if count >= k]
    population_sizes = [comb(counts[index], k) for index in eligible]
    total = sum(population_sizes)
    evaluated = min(total, max_combinations)
    exact = evaluated == total

    selected = {index: [] for index in eligible}
    if exact:
        for speaker_index in eligible:
            selected[speaker_index] = combinations(
                range(counts[speaker_index]), k
            )
        return selected, total, evaluated, exact

    cumulative = []
    running_total = 0
    for population_size in population_sizes:
        running_total += population_size
        cumulative.append(running_total)

    rng = random.Random(_derived_seed(seed, k))
    for global_rank in _sample_unique_ranks(total, evaluated, rng):
        eligible_index = bisect_right(cumulative, global_rank)
        speaker_index = eligible[eligible_index]
        previous = cumulative[eligible_index - 1] if eligible_index else 0
        local_rank = global_rank - previous
        selected[speaker_index].append(
            _unrank_combination(counts[speaker_index], k, local_rank)
        )
    return selected, total, evaluated, exact


def _lid_from_logits(logits, target_index, n_enrolments):
    row_max = logits.max(axis=1)
    log_denominator = row_max + np.log(
        np.exp(logits - row_max[:, np.newaxis]).sum(axis=1)
    )
    target_log_probability = logits[:, target_index] - log_denominator
    return log2(n_enrolments) + target_log_probability / np.log(2.0)


def _lids_for_combinations(
    trial_matrix,
    target_index,
    speaker_combinations,
    k,
    nll_temperature,
    brier_temperature,
):
    """Evaluate the four aggregation methods from identical subset sums."""
    maximum_indexed_values = 2_000_000
    batch_size = max(
        1,
        maximum_indexed_values // (k * trial_matrix.shape[1]),
    )
    iterator = iter(speaker_combinations)
    parts = {method: [] for method in METHODS}
    while batch := list(islice(iterator, batch_size)):
        indices = np.asarray(batch, dtype=int).reshape(-1, k)
        summed_llr = trial_matrix[indices].sum(axis=1)
        logits_by_method = {
            "sum_llr": summed_llr,
            "average_llr": summed_llr / k,
            "temperature_scaled_nll": summed_llr / nll_temperature,
            "temperature_scaled_brier": summed_llr / brier_temperature,
        }
        for method, logits in logits_by_method.items():
            parts[method].append(
                _lid_from_logits(
                    logits,
                    target_index,
                    trial_matrix.shape[1],
                )
            )
    return {
        method: np.concatenate(method_parts)
        for method, method_parts in parts.items()
    }


def _validate_fitted_temperatures(temperatures):
    resolved = {}
    for fit_name in ("nll", "brier"):
        try:
            temperature = float(temperatures[fit_name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Missing finite positive {fit_name} temperature"
            ) from error
        temperature = float(temperature)
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError(
                f"Temperature '{fit_name}' must be finite and "
                "greater than zero"
            )
        resolved[fit_name] = temperature
    return resolved


def combination_lid_populations(
    evidence: LongitudinalEvidence,
    nll_temperature,
    brier_temperature,
    max_combinations=DEFAULT_MAX_COMBINATIONS,
    seed=0,
    max_k=None,
):
    """Evaluate four aggregation methods over same-speaker trial subsets.

    Sampling is uniform over the pooled population of all within-speaker
    subsets. If speaker ``s`` has ``K_s`` available trials, that speaker
    contributes ``C(K_s, k)`` subsets and is skipped whenever ``K_s < k``.
    """
    temperatures = _validate_fitted_temperatures(
        {"nll": nll_temperature, "brier": brier_temperature}
    )
    if max_combinations <= 0:
        raise ValueError("max_combinations must be greater than zero")

    matrices, targets = _speaker_trial_matrices(evidence)
    counts = [len(matrix) for matrix in matrices]
    available_max_k = max(counts)
    if max_k is None:
        max_k = available_max_k
    if max_k <= 0:
        raise ValueError("max_k must be greater than zero")
    max_k = min(int(max_k), available_max_k)

    populations = []
    for k in range(1, max_k + 1):
        selected, total, evaluated, exact = _combination_indices_by_speaker(
            counts,
            k,
            int(max_combinations),
            seed,
        )
        value_parts = {method: [] for method in METHODS}
        for speaker_index, speaker_combinations in selected.items():
            if not exact and not speaker_combinations:
                continue
            speaker_values = _lids_for_combinations(
                matrices[speaker_index],
                targets[speaker_index],
                speaker_combinations,
                k,
                temperatures["nll"],
                temperatures["brier"],
            )
            for method in METHODS:
                value_parts[method].append(speaker_values[method])

        values = {
            method: np.concatenate(value_parts[method])
            for method in METHODS
        }
        if len(values["sum_llr"]) != evaluated:
            raise RuntimeError(
                f"Expected {evaluated} evaluated combinations for k={k}, "
                f"but produced {len(values['sum_llr'])}"
            )
        populations.append(
            CombinationPopulation(
                k=k,
                total_combinations=total,
                eligible_speakers=sum(count >= k for count in counts),
                evaluated_combinations=evaluated,
                is_exact=exact,
                values=values,
            )
        )
    return populations


def _mean_confidence_interval(values, population_size, is_exact):
    mean = float(np.mean(values))
    if is_exact or len(values) < 2:
        return mean, mean

    sample_variance = float(np.var(values, ddof=1))
    sampling_fraction = len(values) / population_size
    finite_population_correction = sqrt(max(0.0, 1.0 - sampling_fraction))
    standard_error = (
        sqrt(sample_variance / len(values)) * finite_population_correction
    )
    return mean - 1.96 * standard_error, mean + 1.96 * standard_error


def _effective_temperature(method, k, temperatures):
    if method == "sum_llr":
        return 1.0
    if method == "average_llr":
        return float(k)
    if method == "temperature_scaled_nll":
        return temperatures["nll"]
    if method == "temperature_scaled_brier":
        return temperatures["brier"]
    raise ValueError(f"Unknown aggregation method: {method}")


def summarize_populations(experiment, temperatures, populations):
    """Create one transparent statistics row per method and evidence count."""
    temperatures = _validate_fitted_temperatures(temperatures)
    rows = []
    for population in populations:
        for method in METHODS:
            values = population.values[method]
            confidence_low, confidence_high = _mean_confidence_interval(
                values,
                population.total_combinations,
                population.is_exact,
            )
            variance_ddof = (
                0 if population.is_exact or len(values) < 2 else 1
            )
            rows.append(
                {
                    "experiment": experiment,
                    "method": method,
                    "method_label": METHOD_STYLES[method]["label"],
                    "k": population.k,
                    "effective_temperature": _effective_temperature(
                        method,
                        population.k,
                        temperatures,
                    ),
                    "eligible_speakers": population.eligible_speakers,
                    "total_combinations": population.total_combinations,
                    "evaluated_combinations": population.evaluated_combinations,
                    "evaluation": "exact" if population.is_exact else "sampled",
                    "mean_LID_bits": float(np.mean(values)),
                    "variance_LID_bits2": float(
                        np.var(values, ddof=variance_ddof)
                    ),
                    "standard_deviation_LID_bits": float(
                        np.std(values, ddof=variance_ddof)
                    ),
                    "mean_95ci_low_bits": confidence_low,
                    "mean_95ci_high_bits": confidence_high,
                    "percentile_2_5_LID_bits": float(np.quantile(values, 0.025)),
                    "median_LID_bits": float(np.median(values)),
                    "percentile_97_5_LID_bits": float(np.quantile(values, 0.975)),
                    "minimum_LID_bits": float(np.min(values)),
                    "maximum_LID_bits": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def _plot_method_axis(ax, method, temperatures, summary):
    style = METHOD_STYLES[method]
    method_summary = summary[summary["method"] == method].sort_values("k")
    positions = method_summary["k"].to_numpy()
    means = method_summary["mean_LID_bits"].to_numpy()
    central_low = method_summary["percentile_2_5_LID_bits"].to_numpy()
    central_high = method_summary["percentile_97_5_LID_bits"].to_numpy()

    ax.fill_betweenx(
        positions,
        central_low,
        central_high,
        color=style["color"],
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(
        means,
        positions,
        color=style["color"],
        linewidth=2.0,
        marker="o",
        markersize=2.8,
        zorder=4,
    )
    ax.axvline(0.0, color="#6B7280", linestyle=":", linewidth=1.0)
    formula = style["formula"].format(**temperatures)
    ax.set_title(f"{style['label']} ({formula})", fontsize=11)
    ax.set_xlabel("Local Information Disclosure (bits)")
    ax.grid(axis="x", color="#D1D5DB", linewidth=0.5, alpha=0.7)
    ax.set_ylim(0.3, positions.max() + 0.7)


def plot_combination_convergence(
    experiment,
    temperatures,
    populations,
    summary,
    output_dir,
    dpi=300,
):
    """Plot the four methods as separate mean-and-interval panels."""
    max_k = populations[-1].k
    figure_height = max(10.0, min(16.0, 8.0 + 0.12 * max_k))
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.4, figure_height),
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.975,
        bottom=0.065,
        top=0.89,
        hspace=0.20,
        wspace=0.08,
    )
    for ax, method in zip(axes.flat, METHODS):
        _plot_method_axis(ax, method, temperatures, summary)

    axes[0, 0].set_ylabel("Number of combined trials (k)")
    axes[1, 0].set_ylabel("Number of combined trials (k)")
    tick_count = min(max_k, 14)
    tick_positions = np.unique(
        np.linspace(1, max_k, tick_count, dtype=int)
    )
    axes[0, 0].set_yticks(tick_positions)
    sampled_n = [
        population.k
        for population in populations
        if not population.is_exact
    ]
    sampling_text = (
        "All combinations evaluated exactly"
        if not sampled_n
        else (
            f"Exact where the population is small; otherwise up to "
            f"{max(population.evaluated_combinations for population in populations):,} "
            "uniformly sampled combinations per k"
        )
    )
    fig.suptitle(
        f"{experiment}: LID convergence as evidence is combined",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.945,
        (
            "Within-speaker subsets only; speakers with fewer than k trials "
            "are skipped. Lines show means; bands show the central 95% of "
            f"subset LID scores. {sampling_text}."
        ),
        ha="center",
        va="top",
        fontsize=9,
        color="#4B5563",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"lid_combination_convergence.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def load_fitted_temperatures(experiment_dir):
    calibration_path = Path(experiment_dir) / "long" / "calibration.json"
    if not calibration_path.is_file():
        raise FileNotFoundError(
            "A fitted temperature requires longitudinal calibration output: "
            f"{calibration_path}"
        )
    calibration = json.loads(calibration_path.read_text())
    try:
        temperatures = {
            "nll": float(
                calibration["methods"]["temperature_scaled"]
                ["fitted_parameters"]["temperature"]
            ),
            "brier": float(
                calibration["methods"]["temperature_scaled_brier"]
                ["fitted_parameters"]["temperature"]
            ),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Could not read the global NLL and Brier temperatures from "
            f"{calibration_path}"
        ) from error
    return _validate_fitted_temperatures(temperatures), calibration_path


def analyse_experiment_convergence(
    experiment_dir,
    output_root,
    max_combinations=DEFAULT_MAX_COMBINATIONS,
    max_k=None,
    seed=0,
    dpi=300,
):
    """Run convergence analysis and write plots, statistics, and metadata."""
    experiment_dir = Path(experiment_dir)
    scores_path = experiment_dir / "scores" / "test_scores.csv"
    evidence = build_longitudinal_evidence(load_trial_scores(scores_path))
    temperatures, temperature_path = load_fitted_temperatures(experiment_dir)
    temperature_source = (
        "development-fitted global NLL and Brier temperatures from "
        f"{temperature_path.name}"
    )

    populations = combination_lid_populations(
        evidence,
        temperatures["nll"],
        temperatures["brier"],
        max_combinations=max_combinations,
        seed=seed,
        max_k=max_k,
    )
    summary = summarize_populations(
        experiment_dir.name,
        temperatures,
        populations,
    )
    output_dir = Path(output_root) / experiment_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "lid_combination_convergence.csv"
    summary.to_csv(summary_path, index=False)
    plot_paths = plot_combination_convergence(
        experiment_dir.name,
        temperatures,
        populations,
        summary,
        output_dir,
        dpi=dpi,
    )

    metadata = {
        "schema_version": 3,
        "experiment": experiment_dir.name,
        "score_source": str(scores_path),
        "temperatures": temperatures,
        "temperature_source": temperature_source,
        "temperature_convention": "softmax(aggregated_LLR / temperature)",
        "aggregation_rules": {
            "sum_llr": "aggregated_LLR = sum_i LLR_i; T = 1",
            "average_llr": "aggregated_LLR = sum_i LLR_i / k; T = k",
            "temperature_scaled_nll": (
                "aggregated_LLR = sum_i LLR_i / T_NLL"
            ),
            "temperature_scaled_brier": (
                "aggregated_LLR = sum_i LLR_i / T_Brier"
            ),
        },
        "combination_population": (
            "All unique unordered k-trial subsets within each trial speaker, "
            "pooled across eligible speakers. Speakers are weighted by their "
            "number of subsets C(K_s, k), where K_s is the number of available "
            "trials for speaker s. A speaker is eligible exactly when K_s >= k; "
            "speakers with fewer than k trials are skipped."
        ),
        "sampling": {
            "seed": seed,
            "maximum_evaluated_combinations_per_n": max_combinations,
            "method": (
                "Exact enumeration when total_combinations <= the maximum; "
                "otherwise a uniform sample without replacement from the "
                "pooled combination population."
            ),
        },
        "k_range": [
            populations[0].k,
            populations[-1].k,
        ],
        "statistics_file": summary_path.name,
        "plot_files": [path.name for path in plot_paths],
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "summary": summary,
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "plot_paths": plot_paths,
        "temperatures": temperatures,
    }


def _display_path(path):
    try:
        return relative_path(path)
    except ValueError:
        return Path(path).resolve()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the distribution and mean of mated LID as every possible "
            "number k of within-speaker trials is combined."
        )
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        metavar="EXPERIMENT",
        help="Only process these experiment directory names.",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=DEFAULT_MAX_COMBINATIONS,
        help=(
            "Maximum exact/sample subset evaluations per k "
            f"(default: {DEFAULT_MAX_COMBINATIONS:,})."
        ),
    )
    parser.add_argument(
        "--max-k",
        type=int,
        help="Stop after this many combined trials (default: available maximum).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for reproducible uniform subset sampling (default: 0).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster plot resolution (default: 300).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "long" / "combination_convergence",
        help="Root output directory.",
    )
    args = parser.parse_args()
    for name in ("max_combinations", "dpi"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_k is not None and args.max_k <= 0:
        parser.error("--max-k must be positive")

    print()
    print(SEPARATOR)
    print("LID combination convergence")
    print(SEPARATOR)
    results = []
    for experiment_dir in iter_experiment_dirs(args.experiments):
        result = analyse_experiment_convergence(
            experiment_dir,
            args.output_dir,
            max_combinations=args.max_combinations,
            max_k=args.max_k,
            seed=args.seed,
            dpi=args.dpi,
        )
        results.append(result["summary"])
        temperature_text = ", ".join(
            f"{name} T={value:.6g}"
            for name, value in result["temperatures"].items()
        )
        print(
            f"{experiment_dir.name}: k=1..{result['summary']['k'].max()}, "
            f"{temperature_text}; wrote "
            f"{_display_path(result['summary_path'].parent)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.output_dir / "summary_table.csv"
    pd.concat(results, ignore_index=True).to_csv(combined_path, index=False)
    print(f"Combined statistics: {_display_path(combined_path)}")
    print(SEPARATOR2)
    print()


if __name__ == "__main__":
    main()
