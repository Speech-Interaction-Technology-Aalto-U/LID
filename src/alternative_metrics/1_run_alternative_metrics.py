import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cllr import compute_cllr, plot_pav_calibration
from eer import compute_eer, plot_eer_histogram
from experiment_paths import scores_path

EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"
OUTPUT_DIR_NAME = "alternative_metrics"
OUTPUT_FILE = "results.json"
SEPARATOR = "-" * 72


def relative_path(path):
    return path.relative_to(PROJECT_ROOT)


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
        "EER": compute_eer(target_scores, non_target_scores),
        "cllr": compute_cllr(df_scores),
    }
    output_json_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"alternative metrics saved in {relative_path(output_json_path)}")
    print(f"saved {relative_path(eer_plot['png_path'])}")
    print(f"saved {relative_path(eer_plot['pdf_path'])}")
    print(f"saved {relative_path(pav_plot['png_path'])}")
    print(f"saved {relative_path(pav_plot['pdf_path'])}")
    print(f"EER = {metrics['EER']:.6g}")
    print(f"cllr = {metrics['cllr']:.6g}")


def process_experiment(experiment_dir):
    test_scores_path = scores_path(experiment_dir, "test")
    if not test_scores_path.is_file():
        raise FileNotFoundError(f"Missing required test scores file: {test_scores_path}")

    output_dir = RESULTS_DIR / experiment_dir.name / OUTPUT_DIR_NAME

    print(SEPARATOR)
    print(f"Experiment: {experiment_dir.name}")
    print(SEPARATOR)
    compute_alternative_metrics(test_scores_path, output_dir)
    print()


def iter_experiment_dirs():
    if not EXPERIMENTS_DIR.is_dir():
        raise FileNotFoundError(f"Missing experiments directory: {EXPERIMENTS_DIR}")

    experiment_dirs = sorted(path for path in EXPERIMENTS_DIR.iterdir() if path.is_dir())
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiment directories found in: {EXPERIMENTS_DIR}")
    return experiment_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute alternative speaker recognition metrics.")
    parser.add_argument(
        "experiment",
        nargs="?",
        help="Experiment name under data/experiments. If omitted, all experiments are processed.",
    )

    args = parser.parse_args()

    print("Computing alternative metrics.")
    print("For each experiment, EER is computed from test similarity scores")
    print("and cllr is computed from raw similarity scores via isotonic regression.")
    print("An EER plot and a PAV calibration plot are also saved")
    print("under results/experiments/{experiment}/alternative_metrics.\n")

    if args.experiment:
        process_experiment(EXPERIMENTS_DIR / args.experiment)
    else:
        for experiment_dir in iter_experiment_dirs():
            process_experiment(experiment_dir)
