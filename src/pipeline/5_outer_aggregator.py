"""
Aggregate trial-level local information disclosure values into final metrics.

For each experiment, this script summarizes the LID values into ALID,
positive/negative disclosure rates, conditional LID averages, and the maximum
observed LID value.

Inputs:
    <experiment>/outputs/local_information_disclosures.csv
    <experiment>/scores/test_scores.csv

Outputs:
    results/experiments/<experiment>/results.json
"""

import json
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from tools.utils import (
    PROJECT_ROOT,
    iter_experiment_dirs,
    lid_path,
    scores_path,
    SEPARATOR,
    SEPARATOR2,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"
METRICS_FILE = "results.json"


def optional_mean(values):
    if len(values) == 0:
        return None
    return float(values.mean())


def aggregate_local_information_disclosures(
    experiment_name,
    lid_csv_path,
    test_scores_path,
    output_json_path,
):
    df_lid = pd.read_csv(lid_csv_path)
    required_columns = {"trial_id", "trial_spk", "p", "LID"}
    missing_columns = required_columns - set(df_lid.columns)
    if missing_columns:
        raise ValueError(
            f"{lid_csv_path} is missing required columns: {sorted(missing_columns)}"
        )
    if df_lid.empty:
        raise ValueError(f"{lid_csv_path} contains no trials")

    df_scores = pd.read_csv(test_scores_path)
    if "enroll_spk" not in df_scores.columns:
        raise ValueError(f"{test_scores_path} is missing required column: enroll_spk")

    n_enrolments = int(df_scores["enroll_spk"].nunique())
    lids = df_lid["LID"]
    positive_lids = lids[lids > 0]
    negative_lids = lids[lids <= 0]
    n_trials = len(lids)

    pdr = len(positive_lids) / n_trials
    ndr = len(negative_lids) / n_trials
    lid_positive = optional_mean(positive_lids)
    lid_negative = optional_mean(negative_lids)
    alid = float(lids.mean())

    metrics = {
        "experiment": experiment_name,
        "n_trials": int(n_trials),
        "n_enrolments": n_enrolments,
        "ALID": alid,
        "PDR": float(pdr),
        "NDR": float(ndr),
        "LID+": lid_positive,
        "LID-": lid_negative,
        "LID_max": float(lids.max()),
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(metrics, indent=2) + "\n")

    return metrics


def process_experiment(experiment_dir):
    lid_csv_path = lid_path(experiment_dir)
    if not lid_csv_path.is_file():
        raise FileNotFoundError(f"Missing required LID file: {lid_csv_path}")
    test_scores_path = scores_path(experiment_dir, "test")
    if not test_scores_path.is_file():
        raise FileNotFoundError(f"Missing required test scores file: {test_scores_path}")

    output_dir = RESULTS_DIR / experiment_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    return aggregate_local_information_disclosures(
        experiment_dir.name,
        lid_csv_path,
        test_scores_path,
        output_dir / METRICS_FILE,
    )


def format_metric(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    print()
    print(SEPARATOR)
    print("STEP 5. Aggregating local information disclosure metrics.")
    print(SEPARATOR)
    print(
        "Calculating ALID, PDR, NDR, LID+, LID-, LID max for each experiment."
    )
    print()

    metrics_by_experiment = []
    for experiment_dir in iter_experiment_dirs():
        metrics_by_experiment.append(process_experiment(experiment_dir))


    print("The privacy results are saved in results/experiments/<experiment>/results.json.")
    print(SEPARATOR2)
    print()
