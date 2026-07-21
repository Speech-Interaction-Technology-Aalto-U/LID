"""
Create local information disclosure values from calibrated test scores.

For each experiment, this script selects the mated enrolment (target speaker) probability for
each test trial and converts it to local information disclosure in bits:
LID = log2(N * p), where N is the number of enrolment profiles.


Inputs:
    <experiment>/scores/test_scores.csv with p and ln_p (for numerical stability, we use ln_p instead of p)

Outputs:
    <experiment>/outputs/local_information_disclosures.csv
"""

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from tools.utils import (
    iter_experiment_dirs,
    lid_path,
    scores_path,
    SEPARATOR,
    SEPARATOR2,
)


def count_text(values, label):
    if len(values) == 1:
        return f"{values.pop()} {label}"
    return f"{min(values)}-{max(values)} {label}"


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

    return {
        "lid_values": len(df_lid),
        "enrolments": n_enrolments,
    }


def process_experiment(experiment_dir):
    test_scores_path = scores_path(experiment_dir, "test")
    if not test_scores_path.is_file():
        raise FileNotFoundError(f"Missing required test scores file: {test_scores_path}")

    return create_local_information_disclosures(test_scores_path, lid_path(experiment_dir))


def print_summary(experiment_summaries):
    lid_text = count_text(
        {summary["lid_values"] for summary in experiment_summaries},
        "LID values",
    )

    print(f"[test]: {lid_text} computed")
    print()
    print("The LID scores are saved in each experiment's outputs/ directory.")


if __name__ == "__main__":
    print()
    print(SEPARATOR)
    print("STEP 4. Creating local information disclosure values for each trial.")
    print(SEPARATOR)

    experiment_summaries = []
    for experiment_dir in iter_experiment_dirs():
        experiment_summaries.append(process_experiment(experiment_dir))

    print_summary(experiment_summaries)
    print(SEPARATOR2)
    print()
