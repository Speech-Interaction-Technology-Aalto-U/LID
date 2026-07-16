import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from experiment_paths import lid_path, scores_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
SEPARATOR = "-" * 72


def relative_path(path):
    return path.relative_to(PROJECT_ROOT)


def create_local_information_disclosures(test_scores_path, output_csv_path):
    df_scores = pd.read_csv(test_scores_path)
    required_columns = {"enroll_spk", "trial_spk", "trial_id", "p", "ln_p"}
    missing_columns = required_columns - set(df_scores.columns)
    if missing_columns:
        raise ValueError(
            f"{test_scores_path} is missing required columns: {sorted(missing_columns)}"
        )

    n_enrolments = df_scores["enroll_spk"].nunique()
    df_mated = df_scores[
        df_scores["enroll_spk"].astype(str) == df_scores["trial_spk"].astype(str)
    ].copy()

    mated_counts = df_mated.groupby("trial_id").size()
    if (mated_counts != 1).any():
        invalid_trial_ids = mated_counts[mated_counts != 1].index[:10].tolist()
        raise ValueError(
            f"Expected exactly one mated enrolment per trial_id. "
            f"Invalid trial_id values: {invalid_trial_ids}"
        )

    missing_mated_trial_ids = sorted(
        set(df_scores["trial_id"].unique()) - set(df_mated["trial_id"].unique())
    )
    if missing_mated_trial_ids:
        raise ValueError(
            f"No mated enrolment found for trial_id values: {missing_mated_trial_ids[:10]}"
        )

    # LID is log2(N * p). Using ln_p avoids multiplying tiny probabilities directly.
    df_lid = df_mated[["trial_id", "trial_spk", "p"]].copy()
    df_lid["LID"] = np.log2(n_enrolments) + df_mated["ln_p"] / np.log(2)
    df_lid = df_lid[["trial_id", "trial_spk", "p", "LID"]]
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_lid.to_csv(output_csv_path, index=False)

    print(
        f"{len(df_lid)} local information disclosure values created "
        f"using {n_enrolments} enrolments"
    )
    print(f"local information disclosures saved in {relative_path(output_csv_path)}")


def process_experiment(experiment_dir):
    test_scores_path = scores_path(experiment_dir, "test")
    if not test_scores_path.is_file():
        raise FileNotFoundError(f"Missing required test scores file: {test_scores_path}")

    print(SEPARATOR)
    print(f"Experiment: {experiment_dir.name}")
    print(SEPARATOR)
    create_local_information_disclosures(test_scores_path, lid_path(experiment_dir))
    print()


def iter_experiment_dirs():
    if not EXPERIMENTS_DIR.is_dir():
        raise FileNotFoundError(f"Missing experiments directory: {EXPERIMENTS_DIR}")

    experiment_dirs = sorted(path for path in EXPERIMENTS_DIR.iterdir() if path.is_dir())
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiment directories found in: {EXPERIMENTS_DIR}")
    return experiment_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create local information disclosure values.")
    parser.add_argument(
        "experiment",
        nargs="?",
        help="Experiment name under data/experiments. If omitted, all experiments are processed.",
    )

    args = parser.parse_args()

    print("Creating local information disclosure values.")
    print("For each test trial, the mated enrolment probability is selected and")
    print("converted to LID bits using LID = log2(N * p).")
    print("The calculation uses ln_p internally for numerical stability.\n")

    if args.experiment:
        process_experiment(EXPERIMENTS_DIR / args.experiment)
    else:
        for experiment_dir in iter_experiment_dirs():
            process_experiment(experiment_dir)
