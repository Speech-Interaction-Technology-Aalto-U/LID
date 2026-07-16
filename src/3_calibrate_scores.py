import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
SPLITS = ("dev", "test")
SEPARATOR = "-" * 72
CALIBRATION_PARAMETERS_FILE = "calibration_parameters.json"


def relative_path(path):
    return path.relative_to(PROJECT_ROOT)


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
    trial_stds = grouped_scores.transform('std', ddof=1) # ddof=0 corresponds to the population variance, but ddof=1 corresponds to the sample variance (where we don't have the full population). We stick to the sample variance because we are just estimating the variance of the population (we don't know it). This is the reason why the variance of each z-normalized row is not exactly 1.
    if (trial_stds == 0).any():
        zero_variance_trials = df_scores.loc[trial_stds == 0, "trial_id"].unique()
        raise ValueError(
            f"Cannot z-normalize scores with zero variance for trial_id values: "
            f"{zero_variance_trials[:10].tolist()}"
        )

    df_scores["z_score"] = (df_scores["score"] - trial_means) / trial_stds
    df_scores.to_csv(scores_csv_path, index=False)

    print(
        f"z_score added for {df_scores['trial_id'].nunique()} trials "
        f"(containing {len(df_scores)} scores)"
    )
    print(f"calibrated scores appended to {relative_path(scores_csv_path)}")


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

    output_json_path.write_text(json.dumps(parameters, indent=2) + "\n")

    print(
        f"dev-only logistic regression fitted on {parameters['n_scores']} scores "
        f"({parameters['n_mated']} mated, {parameters['n_non_mated']} non-mated)"
    )
    print(f"LLR parameters: weight = {weight:.6g}, intercept = {intercept:.6g}")
    print(f"parameters saved in {relative_path(output_json_path)}")
    return parameters


def apply_llr_to_scores(scores_csv_path, parameters):
    df_scores = pd.read_csv(scores_csv_path)
    if "z_score" not in df_scores.columns:
        raise ValueError(f"{scores_csv_path} is missing required column: z_score")

    df_scores["llr"] = parameters["w"] * df_scores["z_score"] + parameters["b"]
    df_scores.to_csv(scores_csv_path, index=False)

    print(f"LLR scores added for {len(df_scores)} scores")
    print(f"LLR scores appended to {relative_path(scores_csv_path)}")


def add_trial_probabilities(scores_csv_path):
    df_scores = pd.read_csv(scores_csv_path)
    if "llr" not in df_scores.columns:
        raise ValueError(f"{scores_csv_path} is missing required column: llr")

    # We calculate log probabilities for numerical stability
    log_denominator = df_scores.groupby("trial_id")["llr"].transform(logsumexp)
    df_scores["ln_p"] = df_scores["llr"] - log_denominator
    df_scores["p"] = np.exp(df_scores["ln_p"])
    df_scores.to_csv(scores_csv_path, index=False)

    print(f"natural-log probabilities added for {df_scores['trial_id'].nunique()} trial utterances")
    print(f"probabilities derived from ln_p and appended to {relative_path(scores_csv_path)}")


def process_experiment(experiment_dir):
    print(SEPARATOR)
    print(f"Experiment: {experiment_dir.name}")
    print(SEPARATOR)
    for split in SPLITS:
        scores_path = experiment_dir / f"{split}_scores.csv"
        if not scores_path.is_file():
            raise FileNotFoundError(f"Missing required scores file: {scores_path}")

        print(f"{split}:")
        add_trial_z_scores(scores_path)
        print()

    print("dev calibration model:")
    parameters = fit_dev_logistic_regression(
        experiment_dir / "dev_scores.csv",
        experiment_dir / CALIBRATION_PARAMETERS_FILE,
    )
    print()

    print("test LLR scores:")
    apply_llr_to_scores(experiment_dir / "test_scores.csv", parameters)
    print()

    print("test probabilities:")
    add_trial_probabilities(experiment_dir / "test_scores.csv")
    print()


def iter_experiment_dirs():
    if not EXPERIMENTS_DIR.is_dir():
        raise FileNotFoundError(f"Missing experiments directory: {EXPERIMENTS_DIR}")

    experiment_dirs = sorted(path for path in EXPERIMENTS_DIR.iterdir() if path.is_dir())
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiment directories found in: {EXPERIMENTS_DIR}")
    return experiment_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate trial scores.")
    parser.add_argument(
        "experiment",
        nargs="?",
        help="Experiment name under data/experiments. If omitted, all experiments are processed.",
    )

    args = parser.parse_args()

    print("Calibrating trial scores.")
    print("For each experiment, dev and test scores are z-normalized per trial_id.")
    print("Then a balanced logistic regression is fit using dev z_score only.")
    print("The learned dev parameters are applied to test z_score to append LLR scores.")
    print("Test LLR scores are converted to per-trial enrolment probabilities.")
    print("Original score values are kept unchanged.\n")

    if args.experiment:
        process_experiment(EXPERIMENTS_DIR / args.experiment)
    else:
        for experiment_dir in iter_experiment_dirs():
            process_experiment(experiment_dir)
