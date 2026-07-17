"""
Calibrate test similarity scores using dev dataset.

This script 
1. z-normalizes scores within each trial,
2. fits a balanced one-dimensional logistic regression on the dev z-scores, 
3. applies the learned calibration to obtain log-likelihood ratios (LLRs) for the test dataset (using z-scores)
4. converts the LLRs into per-trial enrolment probabilities. These probabilities denote the confidence that the attacker has about each of the persons being the target speaker. (all probabilities per trial add up to 1)

Inputs:
    <experiment>/scores/dev_scores.csv
    <experiment>/scores/test_scores.csv

Outputs:
    <experiment>/scores/dev_scores.csv with z_score
    <experiment>/scores/test_scores.csv with z_score, llr, ln_p, and p
    <experiment>/outputs/calibration_parameters.json
"""

import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

from experiment_paths import calibration_parameters_path, scores_path
from tools.utils import iter_experiment_dirs, SEPARATOR, SEPARATOR2

SPLITS = ("dev", "test")


def add_trial_z_scores(scores_csv_path):
    df_scores = pd.read_csv(scores_csv_path)
    required_columns = {"enroll_spk", "trial_spk", "trial_id", "score"}
    missing_columns = required_columns - set(df_scores.columns)
    if missing_columns:
        raise ValueError(
            f"{scores_csv_path} is missing required columns: {sorted(missing_columns)}"
        )

    grouped_scores = df_scores.groupby("trial_id")["score"]
    trial_means = grouped_scores.transform("mean")
    # Use the sample standard deviation because each trial row estimates a larger
    # score distribution; this means normalized rows need not have variance 1.
    trial_stds = grouped_scores.transform("std", ddof=1)
    if (trial_stds == 0).any():
        zero_variance_trials = df_scores.loc[trial_stds == 0, "trial_id"].unique()
        raise ValueError(
            f"Cannot z-normalize scores with zero variance for trial_id values: "
            f"{zero_variance_trials[:10].tolist()}"
        )

    df_scores["z_score"] = (df_scores["score"] - trial_means) / trial_stds
    df_scores.to_csv(scores_csv_path, index=False)

    return {
        "trials": df_scores["trial_id"].nunique(),
        "scores": len(df_scores),
    }


def fit_dev_logistic_regression(dev_scores_path, output_json_path):
    df_scores = pd.read_csv(dev_scores_path)
    required_columns = {"enroll_spk", "trial_spk", "trial_id", "z_score"}
    missing_columns = required_columns - set(df_scores.columns)
    if missing_columns:
        raise ValueError(
            f"{dev_scores_path} is missing required columns: {sorted(missing_columns)}"
        )

    scores_znorm = df_scores["z_score"].to_numpy()
    labels = (
        df_scores["enroll_spk"].astype(str) == df_scores["trial_spk"].astype(str)
    ).astype(int).to_numpy()

    lin_only_znorm = LogisticRegression(class_weight="balanced")
    lin_only_znorm.fit(scores_znorm.reshape(-1, 1), labels)

    weight = float(lin_only_znorm.coef_[0, 0])
    intercept = float(lin_only_znorm.intercept_[0])
    parameters = {
        "name": "z-score linear calibration",
        "class_weight": "balanced",
        "formula": "LLR = w * z_score + b",
        "w": weight,
        "b": intercept,
        "n_scores": int(len(df_scores)),
        "n_mated": int(labels.sum()),
        "n_non_mated": int(len(labels) - labels.sum()),
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(parameters, indent=2) + "\n")

    return parameters


def apply_llr_to_scores(scores_csv_path, parameters):
    df_scores = pd.read_csv(scores_csv_path)
    if "z_score" not in df_scores.columns:
        raise ValueError(f"{scores_csv_path} is missing required column: z_score")

    df_scores["llr"] = parameters["w"] * df_scores["z_score"] + parameters["b"]
    df_scores.to_csv(scores_csv_path, index=False)

    return {"scores": len(df_scores)}


def add_trial_probabilities(scores_csv_path):
    df_scores = pd.read_csv(scores_csv_path)
    if "llr" not in df_scores.columns:
        raise ValueError(f"{scores_csv_path} is missing required column: llr")

    # We calculate log probabilities for numerical stability
    log_denominator = df_scores.groupby("trial_id")["llr"].transform(logsumexp)
    df_scores["ln_p"] = df_scores["llr"] - log_denominator
    df_scores["p"] = np.exp(df_scores["ln_p"])
    df_scores.to_csv(scores_csv_path, index=False)

    return {
        "trials": df_scores["trial_id"].nunique(),
        "scores": len(df_scores),
    }


def process_experiment(experiment_dir):
    z_score_summaries = {}
    for split in SPLITS:
        split_scores_path = scores_path(experiment_dir, split)
        if not split_scores_path.is_file():
            raise FileNotFoundError(f"Missing required scores file: {split_scores_path}")

        z_score_summaries[split] = add_trial_z_scores(split_scores_path)

    parameters = fit_dev_logistic_regression(
        scores_path(experiment_dir, "dev"),
        calibration_parameters_path(experiment_dir),
    )

    llr_summary = apply_llr_to_scores(scores_path(experiment_dir, "test"), parameters)
    probability_summary = add_trial_probabilities(scores_path(experiment_dir, "test"))

    return {
        "z_scores": z_score_summaries,
        "parameters": parameters,
        "llr": llr_summary,
        "probabilities": probability_summary,
    }


def count_text(values, label):
    if len(values) == 1:
        return f"{values.pop()} {label}"
    return f"{min(values)}-{max(values)} {label}"


def print_summary(processed_experiments, experiment_summaries):
    for split in SPLITS:
        summaries = [summary["z_scores"][split] for summary in experiment_summaries]
        score_text = count_text(
            {summary["scores"] for summary in summaries},
            "z_scores",
        )
        print(f"[{split}]: {score_text} computed")
    print()

    print("Computed calibration parameters (for LLR = w*z_score + b):")
    for experiment_name, summary in zip(processed_experiments, experiment_summaries):
        parameters = summary["parameters"]
        print(
            f"{experiment_name}: w={parameters['w']:.6g}, b={parameters['b']:.6g}"
        )
    print()

    llr_counts = {summary["llr"]["scores"] for summary in experiment_summaries}
    probability_counts = {
        summary["probabilities"]["scores"] for summary in experiment_summaries
    }
    test_count_text = count_text(
        llr_counts | probability_counts,
        "LLR scores and probabilities",
    )
    print(f"[test]: {test_count_text} computed")
    print()
    print("Updated the existing score files in each experiment's scores/ directory.")
    print("Saved the calibration parameters in each experiment's outputs/ directory.")


if __name__ == "__main__":
    print()
    print(SEPARATOR)
    print("STEP 3. Calibrating similarity scores.")
    print(SEPARATOR)
    print(
        "Z-normalizing dev/test scores per trial, using dev scores to learn calibration parameters, "
        "and calibrating test scores to LLRs and obtaining probabilities."
    )
    print()

    processed_experiments = []
    experiment_summaries = []
    for experiment_dir in iter_experiment_dirs():
        experiment_summaries.append(process_experiment(experiment_dir))
        processed_experiments.append(experiment_dir.name)

    print_summary(processed_experiments, experiment_summaries)
    print(SEPARATOR2)
    print()
