"""Longitudinal evidence aggregation and leakage analysis."""

from .analysis import (
    LongitudinalEvidence,
    analyse_experiment,
    build_longitudinal_evidence,
    create_longitudinal_summary,
    load_trial_scores,
)
from .calibration import (
    aggregate_logits,
    analyse_calibrated_aggregations,
    build_calibration_dataset,
    calibrate_on_development,
    fit_individual_trial_temperature,
    fit_method_parameter,
    fit_method_parameters,
    load_development_scores,
)
from .similarity import (
    compute_trial_embedding_similarities,
    load_trial_embedding_similarities,
    precompute_experiment_embedding_similarities,
    similarity_path,
)

__all__ = [
    "LongitudinalEvidence",
    "analyse_experiment",
    "build_longitudinal_evidence",
    "create_longitudinal_summary",
    "load_trial_scores",
    "aggregate_logits",
    "analyse_calibrated_aggregations",
    "build_calibration_dataset",
    "calibrate_on_development",
    "fit_individual_trial_temperature",
    "fit_method_parameter",
    "fit_method_parameters",
    "load_development_scores",
    "compute_trial_embedding_similarities",
    "load_trial_embedding_similarities",
    "precompute_experiment_embedding_similarities",
    "similarity_path",
]
