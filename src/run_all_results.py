import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SEPARATOR = "=" * 72

PIPELINE_STEPS = [
    # (
    #     "Generate random embeddings",
    #     SRC_DIR / "tools" / "generate_random_embeddings.py",
    #     lambda args: [],
    # ),
    (
        "Average enrolment embeddings",
        SRC_DIR / "1_average_enrolment_embeddings.py",
        lambda args: experiment_args(args.experiment),
    ),
    (
        "Create scores from embeddings",
        SRC_DIR / "2_create_scores_from_embeddings.py",
        lambda args: experiment_args(args.experiment),
    ),
    (
        "Run alternative metrics",
        SRC_DIR / "alternative_metrics" / "1_run_alternative_metrics.py",
        lambda args: experiment_args(args.experiment),
    ),
    (
        "Calibrate scores",
        SRC_DIR / "3_calibrate_scores.py",
        lambda args: experiment_args(args.experiment),
    ),
    (
        "Create local information disclosures",
        SRC_DIR / "4_inner_aggregator.py",
        lambda args: experiment_args(args.experiment),
    ),
    (
        "Aggregate experiment results",
        SRC_DIR / "5_outer_aggregator.py",
        lambda args: experiment_args(args.experiment),
    ),
    (
        "Plot results",
        SRC_DIR / "6_plot_results.py",
        lambda args: experiment_args(args.experiment),
    ),
]


def experiment_args(experiment):
    return [experiment] if experiment else []


def run_step(step_name, script_path, script_args):
    command = [sys.executable, str(script_path), *script_args]
    print(SEPARATOR)
    print(step_name)
    print(SEPARATOR)
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full local information disclosure pipeline."
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        help="Optional experiment name under data/experiments. If omitted, all experiments are processed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for step_name, script_path, build_args in PIPELINE_STEPS:
        run_step(step_name, script_path, build_args(args))


if __name__ == "__main__":
    main()
