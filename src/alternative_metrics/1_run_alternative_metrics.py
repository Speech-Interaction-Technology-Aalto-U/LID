"""
Compute alternative verification metrics from test similarity scores.

For each experiment, this script computes EER and Cllr from raw test scores and
saves an EER distribution plot, a PAV calibration plot, and a metrics JSON file.

Inputs:
    <experiment>/scores/test_scores.csv

Outputs:
    results/experiments/<experiment>/alternative_metrics/results.json
    results/experiments/<experiment>/alternative_metrics/eer_distribution.png and .pdf
    results/experiments/<experiment>/alternative_metrics/pav_calibration.png and .pdf
"""

import json
from pathlib import Path
import sys
import pandas as pd

from tools.utils import (
    PROJECT_ROOT,
    iter_experiment_dirs,
    scores_path,
    SEPARATOR,
    SEPARATOR2,
)
from cllr import compute_cllr, plot_pav_calibration
from eer import plot_eer_histogram

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    
ALTERNATIVE_METRICS_DIR = PROJECT_ROOT / "src" / "alternative_metrics"
if str(ALTERNATIVE_METRICS_DIR) not in sys.path:
    sys.path.insert(0, str(ALTERNATIVE_METRICS_DIR))



RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"
OUTPUT_DIR_NAME = "alternative_metrics"
OUTPUT_FILE = "results.json"


def split_target_and_non_target_scores(df_scores, score_column):
    if score_column not in df_scores.columns:
        raise ValueError(f"scores dataframe is missing required column: {score_column}")

    labels = df_scores["enroll_spk"].astype(str) == df_scores["trial_spk"].astype(str)
    target_scores = df_scores.loc[labels, score_column].to_numpy()
    non_target_scores = df_scores.loc[~labels, score_column].to_numpy()

    return target_scores, non_target_scores


def compute_alternative_metrics(test_scores_path, output_dir):
    df_scores = pd.read_csv(test_scores_path)
    required_columns = {"enroll_spk", "trial_spk", "score"}
    missing_columns = required_columns - set(df_scores.columns)
    if missing_columns:
        raise ValueError(
            f"{test_scores_path} is missing required columns: {sorted(missing_columns)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = output_dir / OUTPUT_FILE

    target_scores, non_target_scores = split_target_and_non_target_scores(df_scores, "score")

    eer_plot = plot_eer_histogram(
        target_scores,
        non_target_scores,
        out_dir=output_dir,
        filename_base="eer_distribution",
    )

    pav_plot = plot_pav_calibration(
        df_scores,
        out_dir=output_dir,
        filename_base="pav_calibration",
    )

    metrics = {
        "EER": float(eer_plot["eer"]),
        "cllr": compute_cllr(df_scores),
    }
    output_json_path.write_text(json.dumps(metrics, indent=2) + "\n")

    return {
        "metrics": metrics,
        "files": [
            output_json_path,
            eer_plot["png_path"],
            eer_plot["pdf_path"],
            pav_plot["png_path"],
            pav_plot["pdf_path"],
        ],
    }


def process_experiment(experiment_dir):
    test_scores_path = scores_path(experiment_dir, "test")
    if not test_scores_path.is_file():
        raise FileNotFoundError(f"Missing required test scores file: {test_scores_path}")

    output_dir = RESULTS_DIR / experiment_dir.name / OUTPUT_DIR_NAME

    summary = compute_alternative_metrics(test_scores_path, output_dir)
    return {
        "experiment": experiment_dir.name,
        **summary["metrics"],
        "files": len(summary["files"]),
    }


def format_metric(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_results_table(experiment_summaries):
    columns = [
        ("Experiment", "experiment"),
        ("EER", "EER"),
        ("Cllr", "cllr"),
    ]
    rows = [
        [format_metric(summary[key]) for _, key in columns]
        for summary in experiment_summaries
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, (header, _) in enumerate(columns)
    ]

    print(
        "  ".join(
            header.ljust(widths[index])
            for index, (header, _) in enumerate(columns)
        )
    )
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  ".join(
                value.rjust(widths[index]) if index else value.ljust(widths[index])
                for index, value in enumerate(row)
            )
        )


if __name__ == "__main__":
    print()
    print(SEPARATOR)
    print("Computing alternative privacy metrics.")
    print(SEPARATOR)
    print("Computing EER and Cllr from raw test similarity scores. Generating explanatory plots.")
    print()

    experiment_summaries = []
    for experiment_dir in iter_experiment_dirs():
        experiment_summaries.append(process_experiment(experiment_dir))

    total_files = sum(summary["files"] for summary in experiment_summaries)
    print(f"Generated {total_files} alternative metric files.")
    print()
    print_results_table(experiment_summaries)
    print()
    print("Saved under results/experiments/<experiment>/alternative_metrics/.")
    print(SEPARATOR2)
    print()
