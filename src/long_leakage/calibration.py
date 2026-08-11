"""Development-only calibration for longitudinal LLR aggregation."""

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (
    IDENTITY_COLUMNS,
    _stable_softmax,
    aggregate_lid_metrics,
    build_longitudinal_evidence,
    load_trial_scores,
    plot_mated_comparisons,
    validate_trial_scores,
)
from .similarity import (
    load_trial_embedding_similarities,
    precompute_experiment_embedding_similarities,
)


METHOD_ORDER = (
    "sum_llr",
    "average_llr",
    "temperature_scaled",
    "temperature_scaled_brier",
    "count_adjusted",
    "similarity_adjusted",
)
METHOD_METADATA = {
    "sum_llr": {
        "label": "Summed LLR",
        "formula": "z = sum_i LLR_i",
        "parameter_names": (),
        "fit_objective": None,
    },
    "average_llr": {
        "label": "Averaged LLR",
        "formula": "z = sum_i LLR_i / k",
        "parameter_names": (),
        "fit_objective": None,
    },
    "temperature_scaled": {
        "label": "Global temperature (NLL fit)",
        "formula": "z = sum_i LLR_i / temperature",
        "parameter_names": ("temperature",),
        "fit_objective": "multiclass_nll",
    },
    "temperature_scaled_brier": {
        "label": "Global temperature (Brier fit)",
        "formula": "z = sum_i LLR_i / temperature",
        "parameter_names": ("temperature",),
        "fit_objective": "multiclass_brier",
    },
    "count_adjusted": {
        "label": "Count-adjusted",
        "formula": (
            "z = sum_i LLR_i / "
            "[temperature * (1 + rho * (k - 1))]"
        ),
        "parameter_names": ("temperature", "rho"),
        "fit_objective": "multiclass_nll",
    },
    "similarity_adjusted": {
        "label": "Embedding-similarity-adjusted",
        "formula": (
            "z = sum_i LLR_i / "
            "[temperature * (1 + rho * embedding_redundancy)]"
        ),
        "parameter_names": ("temperature", "rho"),
        "fit_objective": "multiclass_nll",
    },
}


@dataclass(frozen=True)
class CalibrationDataset:
    """Speaker-level sufficient statistics used by aggregation methods."""

    sum_llr: np.ndarray
    average_llr: np.ndarray
    counts: np.ndarray
    embedding_redundancy: np.ndarray
    mean_embedding_cosine_similarity: np.ndarray
    mean_positive_embedding_similarity: np.ndarray
    target_indices: np.ndarray
    speaker_ids: tuple[str, ...]
    similarity_source: str = "trial embeddings"

    @property
    def n_speakers(self):
        return len(self.speaker_ids)


@dataclass(frozen=True)
class AggregationResult:
    """Aggregated logits and posterior probabilities for one method."""

    method: str
    parameters: dict[str, float]
    temperatures: np.ndarray
    logits: np.ndarray
    ln_p: np.ndarray
    p: np.ndarray


def load_development_scores(experiment_dir):
    """Load development scores and apply the existing score calibration."""
    experiment_dir = Path(experiment_dir)
    scores_path = experiment_dir / "scores" / "dev_scores.csv"
    if not scores_path.is_file():
        raise FileNotFoundError(f"Missing required development scores: {scores_path}")

    scores = pd.read_csv(
        scores_path,
        dtype={column: "string" for column in IDENTITY_COLUMNS},
    )
    if "llr" not in scores.columns:
        if "z_score" not in scores.columns:
            raise ValueError(f"{scores_path} must contain either llr or z_score")
        parameters_path = (
            experiment_dir / "outputs" / "calibration_parameters.json"
        )
        if not parameters_path.is_file():
            raise FileNotFoundError(
                "Development z-scores require calibration parameters: "
                f"{parameters_path}"
            )
        parameters = json.loads(parameters_path.read_text())
        try:
            weight = float(parameters["w"])
            intercept = float(parameters["b"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{parameters_path} must contain finite numeric w and b values"
            ) from error
        if not np.isfinite((weight, intercept)).all():
            raise ValueError(
                f"{parameters_path} must contain finite numeric w and b values"
            )
        z_scores = pd.to_numeric(scores["z_score"], errors="coerce")
        scores["llr"] = weight * z_scores + intercept

    return validate_trial_scores(scores, source=scores_path)


def _embedding_similarity_statistics(evidence, pairwise):
    if pairwise is None:
        zeros = np.zeros(evidence.n_trial_speakers, dtype=float)
        return zeros, zeros, zeros, "unavailable"

    redundancy = []
    mean_cosine = []
    mean_positive = []
    for speaker_id, n_trials in zip(
        evidence.speaker_ids,
        evidence.speaker_trial_counts,
    ):
        trial_ids = [
            trial_id
            for trial_id, trial_speaker in zip(
                evidence.trial_ids,
                evidence.trial_speakers,
            )
            if trial_speaker == speaker_id
        ]
        speaker_pairs = pairwise[pairwise["trial_spk"] == speaker_id]
        expected_pairs = {
            frozenset(pair)
            for pair in combinations(trial_ids, 2)
        }
        observed_pairs = {
            frozenset((row.trial_id_a, row.trial_id_b))
            for row in speaker_pairs.itertuples(index=False)
        }
        if observed_pairs != expected_pairs or len(speaker_pairs) != len(expected_pairs):
            raise ValueError(
                "Embedding-similarity pairs do not match the longitudinal "
                f"trials for speaker {speaker_id}: expected {len(expected_pairs)}, "
                f"found {len(speaker_pairs)}"
            )

        if len(speaker_pairs) == 0:
            redundancy.append(0.0)
            mean_cosine.append(0.0)
            mean_positive.append(0.0)
            continue
        similarities = speaker_pairs["cosine_similarity"].to_numpy(dtype=float)
        positive = np.clip(similarities, 0.0, 1.0)
        redundancy.append(2.0 * float(positive.sum()) / int(n_trials))
        mean_cosine.append(float(similarities.mean()))
        mean_positive.append(float(positive.mean()))

    return (
        np.asarray(redundancy),
        np.asarray(mean_cosine),
        np.asarray(mean_positive),
        "within-speaker trial-embedding cosine similarity",
    )


def build_calibration_dataset(evidence, embedding_similarities=None):
    """Create targets, counts, and embedding redundancy from evidence."""
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
            "Longitudinal calibration requires a mated enrolment for every "
            f"trial speaker. Missing: {missing[:10]}"
        )

    redundancy, mean_cosine, mean_positive, source = (
        _embedding_similarity_statistics(evidence, embedding_similarities)
    )
    return CalibrationDataset(
        sum_llr=evidence.sum_llr,
        average_llr=evidence.avg_llr,
        counts=evidence.speaker_trial_counts.astype(float),
        embedding_redundancy=redundancy,
        mean_embedding_cosine_similarity=mean_cosine,
        mean_positive_embedding_similarity=mean_positive,
        target_indices=np.asarray(
            [enroll_index[speaker_id] for speaker_id in evidence.speaker_ids],
            dtype=int,
        ),
        speaker_ids=evidence.speaker_ids,
        similarity_source=source,
    )


def _coerce_parameters(method, parameters):
    names = METHOD_METADATA[method]["parameter_names"]
    if not names:
        return {}
    if isinstance(parameters, dict):
        values = {name: float(parameters[name]) for name in names}
    elif len(names) == 1 and parameters is not None:
        values = {names[0]: float(parameters)}
    elif len(names) == 2 and parameters is not None:
        # Backward-compatible shorthand: a scalar specifies rho with tau=1.
        values = {"temperature": 1.0, "rho": float(parameters)}
    else:
        raise ValueError(f"{method} requires parameters: {names}")

    if "temperature" in values and (
        not np.isfinite(values["temperature"])
        or values["temperature"] <= 0
    ):
        raise ValueError(f"{method} requires temperature > 0")
    if "rho" in values and (
        not np.isfinite(values["rho"])
        or not 0.0 <= values["rho"] <= 1.0
    ):
        raise ValueError(f"{method} requires rho in [0, 1]")
    return values


def aggregate_logits(dataset, method, parameters=None):
    """Apply one longitudinal aggregation family to speaker-level evidence."""
    if method not in METHOD_METADATA:
        raise ValueError(f"Unknown aggregation method: {method}")
    parameters = _coerce_parameters(method, parameters)

    if method == "sum_llr":
        temperatures = np.ones(dataset.n_speakers, dtype=float)
    elif method == "average_llr":
        temperatures = dataset.counts.copy()
    elif method in ("temperature_scaled", "temperature_scaled_brier"):
        temperatures = np.full(
            dataset.n_speakers,
            parameters["temperature"],
        )
    elif method == "count_adjusted":
        temperatures = parameters["temperature"] * (
            1.0 + parameters["rho"] * (dataset.counts - 1.0)
        )
    else:
        temperatures = parameters["temperature"] * (
            1.0
            + parameters["rho"] * dataset.embedding_redundancy
        )
    return dataset.sum_llr / temperatures[:, np.newaxis], temperatures


def aggregate_result(dataset, method, parameters=None):
    parameters = _coerce_parameters(method, parameters)
    logits, temperatures = aggregate_logits(dataset, method, parameters)
    ln_p, probabilities = _stable_softmax(logits)
    return AggregationResult(
        method=method,
        parameters=parameters,
        temperatures=temperatures,
        logits=logits,
        ln_p=ln_p,
        p=probabilities,
    )


def _loss_value(logits, target_indices, objective):
    ln_p, probabilities = _stable_softmax(logits)
    rows = np.arange(len(target_indices))
    if objective == "multiclass_nll":
        return float(-ln_p[rows, target_indices].mean())
    if objective == "multiclass_brier":
        one_hot = np.zeros_like(probabilities)
        one_hot[rows, target_indices] = 1.0
        return float(
            np.square(probabilities - one_hot).sum(axis=1).mean()
        )
    raise ValueError(f"Unknown calibration objective: {objective}")


def _multiclass_metrics(logits, target_indices):
    ln_p, probabilities = _stable_softmax(logits)
    rows = np.arange(len(target_indices))
    target_ln_p = ln_p[rows, target_indices]
    target_p = probabilities[rows, target_indices]
    one_hot = np.zeros_like(probabilities)
    one_hot[rows, target_indices] = 1.0
    predictions = np.argmax(probabilities, axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == target_indices

    ece = 0.0
    edges = np.linspace(0.0, 1.0, 6)
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence >= lower) & (
            confidence <= upper if upper == 1.0 else confidence < upper
        )
        if selected.any():
            ece += selected.mean() * abs(
                float(correct[selected].mean())
                - float(confidence[selected].mean())
            )

    return {
        "n_speakers": int(len(target_indices)),
        "multiclass_nll_nats": float(-target_ln_p.mean()),
        "multiclass_brier": float(
            np.square(probabilities - one_hot).sum(axis=1).mean()
        ),
        "top1_accuracy": float(correct.mean()),
        "top1_ece": float(ece),
        "mean_target_probability": float(target_p.mean()),
    }


def _grid_refined_minimum(
    objective,
    lower,
    upper,
    logarithmic=False,
    grid_size=65,
):
    """Find a deterministic bounded scalar minimum without extra dependencies."""
    if logarithmic:
        coordinates = np.linspace(np.log(lower), np.log(upper), grid_size)
        values = np.exp(coordinates)
    else:
        coordinates = np.linspace(lower, upper, grid_size)
        values = coordinates
    losses = np.asarray([objective(float(value)) for value in values])
    best = int(np.argmin(losses))
    if best in (0, len(values) - 1):
        return float(values[best])

    left = float(coordinates[best - 1])
    right = float(coordinates[best + 1])
    inverse_phi = (np.sqrt(5.0) - 1.0) / 2.0
    c = right - inverse_phi * (right - left)
    d = left + inverse_phi * (right - left)

    def coordinate_loss(coordinate):
        value = np.exp(coordinate) if logarithmic else coordinate
        return objective(float(value))

    fc = coordinate_loss(c)
    fd = coordinate_loss(d)
    for _ in range(40):
        if right - left < 1e-8:
            break
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - inverse_phi * (right - left)
            fc = coordinate_loss(c)
        else:
            left, c, fc = c, d, fd
            d = left + inverse_phi * (right - left)
            fd = coordinate_loss(d)
    coordinate = (left + right) / 2.0
    return float(np.exp(coordinate) if logarithmic else coordinate)


def _temperature_upper_bound(dataset):
    return max(100.0, 4.0 * float(dataset.counts.max()))


def _fit_single_temperature(dataset, objective, indices, divisors=None):
    if divisors is None:
        divisors = np.ones(dataset.n_speakers)

    def loss(temperature):
        logits = dataset.sum_llr / (
            temperature * divisors
        )[:, np.newaxis]
        return _loss_value(
            logits[indices],
            dataset.target_indices[indices],
            objective,
        )

    return _grid_refined_minimum(
        loss,
        lower=0.05,
        upper=_temperature_upper_bound(dataset),
        logarithmic=True,
    )


def _fit_temperature_and_rho(dataset, method, objective, indices):
    factors = (
        dataset.counts - 1.0
        if method == "count_adjusted"
        else dataset.embedding_redundancy
    )

    def loss(temperature, rho):
        divisors = 1.0 + rho * factors
        logits = dataset.sum_llr / (
            temperature * divisors
        )[:, np.newaxis]
        return _loss_value(
            logits[indices],
            dataset.target_indices[indices],
            objective,
        )

    profiled_temperatures = {}

    def profiled_loss(rho):
        temperature = _fit_single_temperature(
            dataset,
            objective,
            indices,
            divisors=1.0 + rho * factors,
        )
        profiled_temperatures[float(rho)] = temperature
        return loss(temperature, rho)

    # Profiling T at every rho avoids the endpoint traps that alternating
    # coordinate updates can create when T and rho compensate for one another.
    rho = _grid_refined_minimum(
        profiled_loss,
        lower=0.0,
        upper=1.0,
        grid_size=65,
    )
    temperature = profiled_temperatures.get(float(rho))
    if temperature is None:
        temperature = _fit_single_temperature(
            dataset,
            objective,
            indices,
            divisors=1.0 + rho * factors,
        )
    return {"temperature": float(temperature), "rho": float(rho)}


def fit_method_parameters(dataset, method, indices=None):
    """Fit named method parameters using its declared proper scoring rule."""
    metadata = METHOD_METADATA[method]
    if not metadata["parameter_names"]:
        return {}
    if indices is None:
        indices = np.arange(dataset.n_speakers)
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        raise ValueError("At least one development speaker is required")

    objective = metadata["fit_objective"]
    if len(metadata["parameter_names"]) == 1:
        return {
            "temperature": _fit_single_temperature(
                dataset,
                objective,
                indices,
            )
        }
    return _fit_temperature_and_rho(
        dataset,
        method,
        objective,
        indices,
    )


def fit_method_parameter(dataset, method, indices=None):
    """Backward-compatible scalar interface for one-parameter methods."""
    parameters = fit_method_parameters(dataset, method, indices=indices)
    if len(parameters) == 1:
        return next(iter(parameters.values()))
    return parameters


def _parameter_stability(fold_parameters):
    if not fold_parameters:
        return None
    stability = {}
    for name in fold_parameters[0]:
        values = np.asarray(
            [parameters[name] for parameters in fold_parameters],
            dtype=float,
        )
        stability[name] = {
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "maximum": float(values.max()),
            "percentile_2_5": float(np.quantile(values, 0.025)),
            "percentile_97_5": float(np.quantile(values, 0.975)),
        }
    return stability


def calibrate_on_development(dataset):
    """Fit all methods and compute leave-one-speaker-out diagnostics."""
    summaries = {}
    fold_rows = []
    for method in METHOD_ORDER:
        metadata = METHOD_METADATA[method]
        parameters = fit_method_parameters(dataset, method)
        full_logits, _ = aggregate_logits(dataset, method, parameters)
        full_metrics = _multiclass_metrics(
            full_logits,
            dataset.target_indices,
        )

        if not metadata["parameter_names"]:
            cv_logits = full_logits
            fold_parameters = []
        elif dataset.n_speakers < 2:
            cv_logits = None
            fold_parameters = []
        else:
            cv_logits = np.empty_like(full_logits)
            fold_parameters = []
            all_indices = np.arange(dataset.n_speakers)
            for held_out in range(dataset.n_speakers):
                train_indices = all_indices[all_indices != held_out]
                fold_parameter_values = fit_method_parameters(
                    dataset,
                    method,
                    indices=train_indices,
                )
                fold_logits, _ = aggregate_logits(
                    dataset,
                    method,
                    fold_parameter_values,
                )
                cv_logits[held_out] = fold_logits[held_out]
                fold_parameters.append(fold_parameter_values)
                fold_rows.append(
                    {
                        "method": method,
                        "fit_objective": metadata["fit_objective"],
                        "held_out_speaker": dataset.speaker_ids[held_out],
                        "temperature": fold_parameter_values.get("temperature"),
                        "rho": fold_parameter_values.get("rho"),
                    }
                )

        cv_metrics = (
            _multiclass_metrics(cv_logits, dataset.target_indices)
            if cv_logits is not None
            else None
        )
        fitted_parameter = (
            next(iter(parameters.values()))
            if len(parameters) == 1
            else parameters or None
        )
        summaries[method] = {
            **metadata,
            "fitted_parameter": fitted_parameter,
            "fitted_parameters": parameters,
            "full_development_metrics": full_metrics,
            "leave_one_speaker_out_metrics": cv_metrics,
            "leave_one_speaker_out_parameter_stability": _parameter_stability(
                fold_parameters
            ),
        }
    return summaries, pd.DataFrame(fold_rows)


def _method_scores_dataframe(evidence, result):
    rows = []
    for speaker_index, trial_spk in enumerate(evidence.speaker_ids):
        for enroll_index, enroll_spk in enumerate(evidence.enroll_speakers):
            rows.append(
                {
                    "enroll_spk": enroll_spk,
                    "trial_spk": trial_spk,
                    "n_trials": int(evidence.speaker_trial_counts[speaker_index]),
                    "temperature": result.temperatures[speaker_index],
                    "llr": result.logits[speaker_index, enroll_index],
                    "ln_p": result.ln_p[speaker_index, enroll_index],
                    "p": result.p[speaker_index, enroll_index],
                }
            )
    return pd.DataFrame(rows)


def _method_mated_dataframe(evidence, dataset, result):
    before = evidence.mated_dataframe()
    before = before[before["stage"] == "before"].copy()
    before["temperature"] = 1.0
    before["n_trials"] = 1

    rows = np.arange(dataset.n_speakers)
    target_ln_p = result.ln_p[rows, dataset.target_indices]
    after = pd.DataFrame(
        {
            "stage": "after",
            "trial_id": [f"combined:{speaker}" for speaker in evidence.speaker_ids],
            "trial_spk": evidence.speaker_ids,
            "p": result.p[rows, dataset.target_indices],
            "LID": np.log2(evidence.n_enrolments) + target_ln_p / np.log(2.0),
            "temperature": result.temperatures,
            "n_trials": evidence.speaker_trial_counts,
        }
    )
    return pd.concat((before, after), ignore_index=True)


def _test_metrics(experiment_name, evidence, mated, method):
    lids = mated.loc[mated["stage"] == "after", "LID"].to_numpy()
    return aggregate_lid_metrics(
        experiment_name,
        lids,
        evidence.n_enrolments,
        aggregation=method,
    )


def _write_new_method_outputs(
    experiment_name,
    evidence,
    dataset,
    result,
    calibration,
    output_dir,
    dpi,
):
    method_dir = output_dir / result.method
    method_dir.mkdir(parents=True, exist_ok=True)
    scores = _method_scores_dataframe(evidence, result)
    mated = _method_mated_dataframe(evidence, dataset, result)
    scores.to_csv(method_dir / "scores.csv", index=False)
    mated.to_csv(method_dir / "mated_probabilities.csv", index=False)
    summary = {
        "aggregation": result.method,
        "formula": METHOD_METADATA[result.method]["formula"],
        "fitted_on": "development speakers only",
        "calibration": calibration,
        "evaluation": _test_metrics(
            experiment_name,
            evidence,
            mated,
            result.method,
        ),
    }
    (method_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return plot_mated_comparisons(
        evidence,
        mated,
        method_dir,
        dpi=dpi,
        after_stage="after",
        after_label=METHOD_METADATA[result.method]["label"],
    )


def _method_comparison_dataframe(evidence, dataset, results):
    raw_mated = evidence.mated_dataframe()
    raw_mated = raw_mated[raw_mated["stage"] == "before"]
    direct = raw_mated.groupby("trial_spk", sort=False)[["p", "LID"]].mean()
    rows = []
    for speaker_index, speaker_id in enumerate(evidence.speaker_ids):
        rows.append(
            {
                "method": "direct_mean",
                "method_label": "Direct per-speaker mean",
                "trial_spk": speaker_id,
                "n_trials": int(dataset.counts[speaker_index]),
                "temperature": np.nan,
                "embedding_redundancy": dataset.embedding_redundancy[speaker_index],
                "p": direct.loc[speaker_id, "p"],
                "LID": direct.loc[speaker_id, "LID"],
            }
        )
        target = dataset.target_indices[speaker_index]
        for method in METHOD_ORDER:
            result = results[method]
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_METADATA[method]["label"],
                    "trial_spk": speaker_id,
                    "n_trials": int(dataset.counts[speaker_index]),
                    "temperature": result.temperatures[speaker_index],
                    "embedding_redundancy": dataset.embedding_redundancy[
                        speaker_index
                    ],
                    "p": result.p[speaker_index, target],
                    "LID": np.log2(evidence.n_enrolments)
                    + result.ln_p[speaker_index, target] / np.log(2.0),
                }
            )
    return pd.DataFrame(rows)


def _diagnostics_dataframe(calibration):
    rows = []
    for method in METHOD_ORDER:
        summary = calibration[method]
        row = {
            "method": method,
            "method_label": summary["label"],
            "fit_objective": summary["fit_objective"],
            "fitted_temperature": summary["fitted_parameters"].get("temperature"),
            "fitted_rho": summary["fitted_parameters"].get("rho"),
        }
        for prefix, metrics in (
            ("development_fit", summary["full_development_metrics"]),
            ("development_loo", summary["leave_one_speaker_out_metrics"]),
        ):
            if metrics is not None:
                for metric_name, value in metrics.items():
                    row[f"{prefix}_{metric_name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _temperature_sweep(dataset, fitted_temperatures):
    maximum = max(
        _temperature_upper_bound(dataset),
        2.0 * max(fitted_temperatures),
    )
    temperatures = np.geomspace(0.05, maximum, 241)
    rows = []
    for temperature in temperatures:
        logits, _ = aggregate_logits(
            dataset,
            "temperature_scaled",
            {"temperature": float(temperature)},
        )
        metrics = _multiclass_metrics(logits, dataset.target_indices)
        rows.append(
            {
                "temperature": temperature,
                "development_multiclass_nll_nats": metrics[
                    "multiclass_nll_nats"
                ],
                "development_multiclass_brier": metrics["multiclass_brier"],
            }
        )
    return pd.DataFrame(rows)


def analyse_calibrated_aggregations(experiment_dir, dpi=300):
    """Fit on development data, evaluate on test data, and write artifacts."""
    experiment_dir = Path(experiment_dir)
    test_scores_path = experiment_dir / "scores" / "test_scores.csv"
    if not test_scores_path.is_file():
        raise FileNotFoundError(f"Missing required test scores: {test_scores_path}")

    similarity_summaries = precompute_experiment_embedding_similarities(
        experiment_dir
    )
    development_evidence = build_longitudinal_evidence(
        load_development_scores(experiment_dir)
    )
    test_evidence = build_longitudinal_evidence(load_trial_scores(test_scores_path))
    development = build_calibration_dataset(
        development_evidence,
        load_trial_embedding_similarities(experiment_dir, "dev"),
    )
    test = build_calibration_dataset(
        test_evidence,
        load_trial_embedding_similarities(experiment_dir, "test"),
    )
    if development_evidence.n_enrolments != test_evidence.n_enrolments:
        raise ValueError(
            "Development and test splits must have the same number of enrolments "
            "for longitudinal calibration"
        )

    method_calibration, folds = calibrate_on_development(development)
    results = {
        method: aggregate_result(
            test,
            method,
            method_calibration[method]["fitted_parameters"],
        )
        for method in METHOD_ORDER
    }

    output_dir = experiment_dir / "long"
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_payload = {
        "schema_version": 2,
        "experiment": experiment_dir.name,
        "temperature_convention": "softmax(sum(LLR) / effective_temperature)",
        "data_usage": {
            "parameter_fitting": "development speakers only",
            "model_diagnostics": "leave-one-development-speaker-out",
            "test_usage": "final evaluation and sensitivity display only",
            "automatic_test_based_model_selection": False,
        },
        "development": {
            "n_speakers": development.n_speakers,
            "min_trials_per_speaker": int(development.counts.min()),
            "max_trials_per_speaker": int(development.counts.max()),
        },
        "embedding_similarity": {
            "source": development.similarity_source,
            "formula": (
                "R_s = (2/k_s) * sum_{i<j} "
                "clip(cosine(e_i, e_j), 0, 1)"
            ),
            "precomputed_files": {
                split: str(summary["path"].relative_to(experiment_dir))
                for split, summary in similarity_summaries.items()
            },
        },
        "methods": method_calibration,
    }
    calibration_path = output_dir / "calibration.json"
    calibration_path.write_text(json.dumps(calibration_payload, indent=2) + "\n")

    diagnostics_path = output_dir / "calibration_diagnostics.csv"
    _diagnostics_dataframe(method_calibration).to_csv(diagnostics_path, index=False)
    folds_path = output_dir / "calibration_folds.csv"
    folds.to_csv(folds_path, index=False)
    fitted_temperatures = [
        summary["fitted_parameters"]["temperature"]
        for summary in method_calibration.values()
        if "temperature" in summary["fitted_parameters"]
    ]
    sweep = _temperature_sweep(development, fitted_temperatures)
    sweep_path = output_dir / "temperature_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    comparison = _method_comparison_dataframe(test_evidence, test, results)
    comparison_path = output_dir / "method_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    redundancy_path = output_dir / "speaker_redundancy.csv"
    pd.DataFrame(
        {
            "trial_spk": test_evidence.speaker_ids,
            "n_trials": test.counts.astype(int),
            "mean_embedding_cosine_similarity": (
                test.mean_embedding_cosine_similarity
            ),
            "mean_positive_embedding_similarity": (
                test.mean_positive_embedding_similarity
            ),
            "embedding_redundancy_R": test.embedding_redundancy,
            "count_adjusted_effective_temperature": results[
                "count_adjusted"
            ].temperatures,
            "similarity_adjusted_effective_temperature": results[
                "similarity_adjusted"
            ].temperatures,
        }
    ).to_csv(redundancy_path, index=False)

    plot_paths = []
    for method in (
        "temperature_scaled",
        "temperature_scaled_brier",
        "count_adjusted",
        "similarity_adjusted",
    ):
        plot_paths.extend(
            _write_new_method_outputs(
                experiment_dir.name,
                test_evidence,
                test,
                results[method],
                method_calibration[method],
                output_dir,
                dpi,
            )
        )

    from .report import write_interactive_report

    report_path = output_dir / "interactive_longitudinal.html"
    write_interactive_report(
        report_path,
        experiment_dir.name,
        test_evidence,
        test,
        results,
        comparison,
        calibration_payload,
        sweep,
    )

    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        summary["calibrated_aggregation"] = {
            "calibration_path": calibration_path.name,
            "method_comparison_path": comparison_path.name,
            "interactive_report_path": report_path.name,
            "development_only_fitting": True,
            "embedding_similarity_files": {
                split: str(summary["path"].relative_to(experiment_dir))
                for split, summary in similarity_summaries.items()
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    return {
        "calibration_path": calibration_path,
        "diagnostics_path": diagnostics_path,
        "folds_path": folds_path,
        "temperature_sweep_path": sweep_path,
        "method_comparison_path": comparison_path,
        "redundancy_path": redundancy_path,
        "report_path": report_path,
        "plot_paths": plot_paths,
        "similarity_paths": {
            split: summary["path"]
            for split, summary in similarity_summaries.items()
        },
        "fitted_temperature": method_calibration[
            "temperature_scaled"
        ]["fitted_parameters"]["temperature"],
        "fitted_brier_temperature": method_calibration[
            "temperature_scaled_brier"
        ]["fitted_parameters"]["temperature"],
        "fitted_count_parameters": method_calibration[
            "count_adjusted"
        ]["fitted_parameters"],
        "fitted_similarity_parameters": method_calibration[
            "similarity_adjusted"
        ]["fitted_parameters"],
    }
