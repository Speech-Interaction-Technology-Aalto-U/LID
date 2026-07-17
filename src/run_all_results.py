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
    ),
    (
        "Create scores from embeddings",
        SRC_DIR / "2_create_scores_from_embeddings.py",
    ),
    (
        "Run alternative metrics",
        SRC_DIR / "alternative_metrics" / "1_run_alternative_metrics.py",
    ),
    (
        "Calibrate scores",
        SRC_DIR / "3_calibrate_scores.py",
    ),
    (
        "Create local information disclosures",
        SRC_DIR / "4_inner_aggregator.py",
    ),
    (
        "Aggregate experiment results",
        SRC_DIR / "5_outer_aggregator.py",
    ),
    (
        "Plot results",
        SRC_DIR / "6_create_plots_and_summaries.py",
    ),
]


def run_step(step_name, script_path):
    command = [sys.executable, str(script_path)]
    # print(SEPARATOR)
    # print(step_name)
    # print(SEPARATOR)
    # print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print()


def main():
    for step_name, script_path in PIPELINE_STEPS:
        run_step(step_name, script_path)


if __name__ == "__main__":
    main()
