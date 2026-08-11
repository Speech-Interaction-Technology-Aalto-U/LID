"""Development-only calibration for longitudinal LLR aggregation."""

from dataclasses import dataclass
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


METHOD_ORDER = (
    "sum_llr",
    "average_llr",
    "temperature_scaled",
    "count_adjusted",
    "similarity_adjusted",
)
METHOD_METADATA = {
    "sum_llr": {
        "label": "Summed LLR",
        "formula": "z = sum_i LLR_i",
        "parameter": None,
    },
    "average_llr": {
        "label": "Averaged LLR",
        "formula": "z = sum_i LLR_i / k",
        "parameter": None,
    },
    "temperature_scaled": {
        "label": "Fitted temperature",
        "formula": "z = sum_i LLR_i / T",
        "parameter": "temperature",
    },
    "count_adjusted": {
        "label": "Count-adjusted",
        "formula": "z = sum_i LLR_i / (1 + rho * (k - 1))",
        "parameter": "rho",
    },
    "similarity_adjusted": {
        "label": "Similarity-adjusted",
        "formula": "z = sum_i LLR_i / (1 + rho * R)",
        "parameter": "rho",
    },
}


@dataclass(frozen=True)
class CalibrationDataset:
    """Speaker-level sufficient statistics used by every aggregation method."""

    sum_llr: np.ndarray
    average_llr: np.ndarray
    counts: np.ndarray
    redundancy: np.ndarray
    mean_positive_similarity: np.ndarray
    target_indices: np.ndarray
    speaker_ids: tuple[str, ...]

    @property
    def n_speakers(self):
        return len(self.speaker_ids)


@dataclass(frozen=True)
class AggregationResult:
    """Aggregated logits and posterior probabilities for one method."""

    method: str
    parameter: float | None
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
            raise ValueError(
                f"{scores_path} must contain either llr or z_score"
            )
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


def _split_speaker_matrices(evidence):
    split_points = np.cumsum(evidence.speaker_trial_counts)[:-1]
    return np.split(evidence.raw_llr, split_points)


def _similarity_statistics(matrix):
    """Return a design-effect redundancy and mean positive cosine similarity."""
    n_trials = len(matrix)
    if n_trials < 2:
        return 0.0, 0.0

    centered = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = np.divide(
        centered,
        norms,
        out=np.zeros_like(centered),
        where=norms > 0,
    )
    similarities = normalized @ normalized.T
    upper = np.maximum(
        similarities[np.triu_indices(n_trials, k=1)],
        0.0,
    )
    pair_sum = float(upper.sum())
    redundancy = 2.0 * pair_sum / n_trials
    mean_similarity = 2.0 * pair_sum / (n_trials * (n_trials - 1))
    return redundancy, mean_similarity


def build_calibration_dataset(evidence):
    """Create target indices, counts, and correlation proxies from evidence."""
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

    statistics = [
        _similarity_statistics(matrix)
        for matrix in _split_speaker_matrices(evidence)
    ]
    redundancy, mean_similarity = zip(*statistics)
    return CalibrationDataset(
        sum_llr=evidence.sum_llr,
        average_llr=evidence.avg_llr,
        counts=evidence.speaker_trial_counts.astype(float),
        redundancy=np.asarray(redundancy, dtype=float),
        mean_positive_similarity=np.asarray(mean_similarity, dtype=float),
        target_indices=np.asarray(
            [enroll_index[speaker_id] for speaker_id in evidence.speaker_ids],
            dtype=int,
        ),
        speaker_ids=evidence.speaker_ids,
    )


def aggregate_logits(dataset, method, parameter=None):
    """Apply one longitudinal aggregation family to speaker-level evidence."""
    if method not in METHOD_METADATA:
        raise ValueError(f"Unknown aggregation method: {method}")

    if method == "sum_llr":
        temperatures = np.ones(dataset.n_speakers, dtype=float)
    elif method == "average_llr":
        temperatures = dataset.counts.copy()
    elif method == "temperature_scaled":
        if parameter is None or not np.isfinite(parameter) or parameter <= 0:
            raise ValueError("temperature_scaled requires a positive temperature")
        temperatures = np.full(dataset.n_speakers, float(parameter))
    elif method == "count_adjusted":
        if parameter is None or not 0.0 <= parameter <= 1.0:
            raise ValueError("count_adjusted requires rho in [0, 1]")
        temperatures = 1.0 + float(parameter) * (dataset.counts - 1.0)
    else:
        if parameter is None or not 0.0 <= parameter <= 1.0:
            raise ValueError("similarity_adjusted requires rho in [0, 1]")
        temperatures = 1.0 + float(parameter) * dataset.redundancy

    return dataset.sum_llr / temperatures[:, np.newaxis], temperatures


def aggregate_result(dataset, method, parameter=None):
    logits, temperatures = aggregate_logits(dataset, method, parameter)
    ln_p, probabilities = _stable_softmax(logits)
    return AggregationResult(
        method=method,
        parameter=parameter,
        temperatures=temperatures,
        logits=logits,
        ln_p=ln_p,
        p=probabilities,
    )


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
    for lower, upper in zip(np.linspace(0.0, 1.0, 6)[:-1], np.linspace(0.0, 1.0, 6)[1:]):
        if upper == 1.0:
            selected = (confidence >= lower) & (confidence <= upper)
        else:
            selected = (confidence >= lower) & (confidence < upper)
        if selected.any():
            ece += selected.mean() * abs(
                float(correct[selected].mean())
                - float(confidence[selected].mean())
            )

    return {
        "n_speakers": int(len(target_indices)),
        "multiclass_nll_nats": float(-target_ln_p.mean()),
        "multiclass_brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "top1_accuracy": float(correct.mean()),
        "top1_ece": float(ece),
        "mean_target_probability": float(target_p.mean()),
    }


def _grid_refined_minimum(objective, lower, upper, logarithmic=False):
    """Find a deterministic bounded scalar minimum without extra dependencies."""
    if logarithmic:
        grid_coordinates = np.linspace(np.log(lower), np.log(upper), 321)
        values = np.exp(grid_coordinates)
    else:
        grid_coordinates = np.linspace(lower, upper, 321)
        values = grid_coordinates
    losses = np.asarray([objective(float(value)) for value in values])
    best_index = int(np.argmin(losses))
    if best_index in (0, len(values) - 1):
        return float(values[best_index])

    left = float(grid_coordinates[best_index - 1])
    right = float(grid_coordinates[best_index + 1])
    inverse_phi = (np.sqrt(5.0) - 1.0) / 2.0
    c = right - inverse_phi * (right - left)
    d = left + inverse_phi * (right - left)

    def coordinate_loss(coordinate):
        value = np.exp(coordinate) if logarithmic else coordinate
        return objective(float(value))

    fc = coordinate_loss(c)
    fd = coordinate_loss(d)
    for _ in range(64):
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


def fit_method_parameter(dataset, method, indices=None):
    """Fit one scalar aggregation parameter by multiclass log loss."""
    parameter_name = METHOD_METADATA[method]["parameter"]
    if parameter_name is None:
        return None

    if indices is None:
        indices = np.arange(dataset.n_speakers)
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        raise ValueError("At least one development speaker is required")

    def objective(parameter):
        logits, _ = aggregate_logits(dataset, method, parameter)
        return _multiclass_metrics(
            logits[indices],
            dataset.target_indices[indices],
        )["multiclass_nll_nats"]

    if method == "temperature_scaled":
        maximum = max(100.0, 4.0 * float(dataset.counts.max()))
        return _grid_refined_minimum(
            objective,
            lower=0.05,
            upper=maximum,
            logarithmic=True,
        )
    return _grid_refined_minimum(
        objective,
        lower=0.0,
        upper=1.0,
    )


def _parameter_stability(values):
    if not values:
        return None
    values = np.asarray(values, dtype=float)
    return {
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
        "percentile_2_5": float(np.quantile(values, 0.025)),
        "percentile_97_5": float(np.quantile(values, 0.975)),
    }


def calibrate_on_development(dataset):
    """Fit all methods and compute leave-one-speaker-out diagnostics."""
    summaries = {}
    fold_rows = []
    for method in METHOD_ORDER:
        parameter_name = METHOD_METADATA[method]["parameter"]
        parameter = fit_method_parameter(dataset, method)
        full_logits, _ = aggregate_logits(dataset, method, parameter)
        full_metrics = _multiclass_metrics(
            full_logits,
            dataset.target_indices,
        )

        if parameter_name is None:
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
                fold_parameter = fit_method_parameter(
                    dataset,
                    method,
                    indices=train_indices,
                )
                fold_logits, _ = aggregate_logits(
                    dataset,
                    method,
                    fold_parameter,
                )
                cv_logits[held_out] = fold_logits[held_out]
                fold_parameters.append(fold_parameter)
                fold_rows.append(
                    {
                        "method": method,
                        "held_out_speaker": dataset.speaker_ids[held_out],
                        "parameter": fold_parameter,
                    }
                )

        cv_metrics = (
            _multiclass_metrics(cv_logits, dataset.target_indices)
            if cv_logits is not None
            else None
        )
        summaries[method] = {
            **METHOD_METADATA[method],
            "fitted_parameter": parameter,
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


def _method_mated_dataframe(evidence, result):
    before = evidence.mated_dataframe()
    before = before[before["stage"] == "before"].copy()
    before["temperature"] = 1.0
    before["n_trials"] = 1

    dataset = build_calibration_dataset(evidence)
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
    result,
    calibration,
    output_dir,
    dpi,
):
    method_dir = output_dir / result.method
    method_dir.mkdir(parents=True, exist_ok=True)
    scores = _method_scores_dataframe(evidence, result)
    mated = _method_mated_dataframe(evidence, result)
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


def _method_comparison_dataframe(evidence, results):
    dataset = build_calibration_dataset(evidence)
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
                "redundancy": dataset.redundancy[speaker_index],
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
                    "redundancy": dataset.redundancy[speaker_index],
                    "p": result.p[speaker_index, target],
                    "LID": np.log2(evidence.n_enrolments)
                    + result.ln_p[speaker_index, target] / np.log(2.0),
                }
            )
    return pd.DataFrame(rows)


def _diagnostics_dataframe(calibration):
    rows = []
    for method in METHOD_ORDER:
        method_summary = calibration[method]
        row = {
            "method": method,
            "method_label": method_summary["label"],
            "parameter_name": method_summary["parameter"],
            "fitted_parameter": method_summary["fitted_parameter"],
        }
        for prefix, metrics in (
            ("development_fit", method_summary["full_development_metrics"]),
            ("development_loo", method_summary["leave_one_speaker_out_metrics"]),
        ):
            if metrics is not None:
                for metric_name, value in metrics.items():
                    row[f"{prefix}_{metric_name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _temperature_sweep(dataset, fitted_temperature):
    maximum = max(
        100.0,
        4.0 * float(dataset.counts.max()),
        2.0 * float(fitted_temperature),
    )
    temperatures = np.geomspace(0.05, maximum, 241)
    rows = []
    for temperature in temperatures:
        logits, _ = aggregate_logits(
            dataset,
            "temperature_scaled",
            float(temperature),
        )
        rows.append(
            {
                "temperature": temperature,
                "development_multiclass_nll_nats": _multiclass_metrics(
                    logits,
                    dataset.target_indices,
                )["multiclass_nll_nats"],
            }
        )
    return pd.DataFrame(rows)


def analyse_calibrated_aggregations(experiment_dir, dpi=300):
    """Fit on development data, evaluate on test data, and write artifacts."""
    experiment_dir = Path(experiment_dir)
    test_scores_path = experiment_dir / "scores" / "test_scores.csv"
    if not test_scores_path.is_file():
        raise FileNotFoundError(f"Missing required test scores: {test_scores_path}")

    development_evidence = build_longitudinal_evidence(
        load_development_scores(experiment_dir)
    )
    test_evidence = build_longitudinal_evidence(load_trial_scores(test_scores_path))
    development = build_calibration_dataset(development_evidence)
    test = build_calibration_dataset(test_evidence)
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
            method_calibration[method]["fitted_parameter"],
        )
        for method in METHOD_ORDER
    }

    output_dir = experiment_dir / "long"
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_payload = {
        "schema_version": 1,
        "experiment": experiment_dir.name,
        "temperature_convention": "softmax(sum(LLR) / T)",
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
        "methods": method_calibration,
    }
    calibration_path = output_dir / "calibration.json"
    calibration_path.write_text(json.dumps(calibration_payload, indent=2) + "\n")

    diagnostics_path = output_dir / "calibration_diagnostics.csv"
    _diagnostics_dataframe(method_calibration).to_csv(diagnostics_path, index=False)
    folds_path = output_dir / "calibration_folds.csv"
    folds.to_csv(folds_path, index=False)
    sweep = _temperature_sweep(
        development,
        method_calibration["temperature_scaled"]["fitted_parameter"],
    )
    sweep_path = output_dir / "temperature_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    comparison = _method_comparison_dataframe(test_evidence, results)
    comparison_path = output_dir / "method_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    redundancy_path = output_dir / "speaker_redundancy.csv"
    pd.DataFrame(
        {
            "trial_spk": test_evidence.speaker_ids,
            "n_trials": test.counts.astype(int),
            "mean_positive_cosine_similarity": test.mean_positive_similarity,
            "redundancy_R": test.redundancy,
            "count_adjusted_temperature": results[
                "count_adjusted"
            ].temperatures,
            "similarity_adjusted_temperature": results[
                "similarity_adjusted"
            ].temperatures,
        }
    ).to_csv(redundancy_path, index=False)

    plot_paths = []
    for method in (
        "temperature_scaled",
        "count_adjusted",
        "similarity_adjusted",
    ):
        plot_paths.extend(
            _write_new_method_outputs(
                experiment_dir.name,
                test_evidence,
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
        "fitted_temperature": method_calibration[
            "temperature_scaled"
        ]["fitted_parameter"],
    }
