import argparse
import json
import pandas as pd
from pathlib import Path

from experiment_paths import lid_path, scores_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"
METRICS_FILE = "results.json"
SEPARATOR = "-" * 72


def relative_path(path):
    return path.relative_to(PROJECT_ROOT)


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

    print(f"{n_trials} trial-level LID values aggregated")
    print(f"ALID = {alid:.6g} bits")
    print(f"PDR = {pdr:.6g}, NDR = {ndr:.6g}")
    print(f"metrics saved in {relative_path(output_json_path)}")


def process_experiment(experiment_dir):
    lid_csv_path = lid_path(experiment_dir)
    if not lid_csv_path.is_file():
        raise FileNotFoundError(f"Missing required LID file: {lid_csv_path}")
    test_scores_path = scores_path(experiment_dir, "test")
    if not test_scores_path.is_file():
        raise FileNotFoundError(f"Missing required test scores file: {test_scores_path}")

    output_dir = RESULTS_DIR / experiment_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(SEPARATOR)
    print(f"Experiment: {experiment_dir.name}")
    print(SEPARATOR)
    aggregate_local_information_disclosures(
        experiment_dir.name,
        lid_csv_path,
        test_scores_path,
        output_dir / METRICS_FILE,
    )
    print()


def iter_experiment_dirs():
    if not EXPERIMENTS_DIR.is_dir():
        raise FileNotFoundError(f"Missing experiments directory: {EXPERIMENTS_DIR}")

    experiment_dirs = sorted(path for path in EXPERIMENTS_DIR.iterdir() if path.is_dir())
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiment directories found in: {EXPERIMENTS_DIR}")
    return experiment_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate local information disclosure metrics."
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        help="Experiment name under data/experiments. If omitted, all experiments are processed.",
    )

    args = parser.parse_args()

    print("Aggregating local information disclosure metrics.")
    print("For each experiment, trial-level LID bits are summarized into ALID,")
    print("positive/negative disclosure rates, and conditional disclosure averages.\n")

    if args.experiment:
        process_experiment(EXPERIMENTS_DIR / args.experiment)
    else:
        for experiment_dir in iter_experiment_dirs():
            process_experiment(experiment_dir)
