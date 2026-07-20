import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
SEPARATOR = "=" * 72

EMBEDDING_STEPS = [
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
]

PIPELINE_STEPS = [
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


def experiment_input_modes():
    if not EXPERIMENTS_DIR.is_dir():
        raise FileNotFoundError(f"Missing experiments directory: {EXPERIMENTS_DIR}")

    experiment_dirs = sorted(path for path in EXPERIMENTS_DIR.iterdir() if path.is_dir())
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiment directories found in: {EXPERIMENTS_DIR}")

    embedding_experiments = []
    score_experiments = []
    incomplete_experiments = []
    for experiment_dir in experiment_dirs:
        embeddings_path = experiment_dir / "embeddings" / "embeddings.parquet"
        scores_dir = experiment_dir / "scores"
        dev_scores_path = scores_dir / "dev_scores.csv"
        test_scores_path = scores_dir / "test_scores.csv"

        if embeddings_path.is_file():
            embedding_experiments.append(experiment_dir.name)
        elif dev_scores_path.is_file() and test_scores_path.is_file():
            score_experiments.append(experiment_dir.name)
        else:
            missing_inputs = []
            if not embeddings_path.is_file():
                missing_inputs.append(str(embeddings_path.relative_to(PROJECT_ROOT)))
            if not dev_scores_path.is_file():
                missing_inputs.append(str(dev_scores_path.relative_to(PROJECT_ROOT)))
            if not test_scores_path.is_file():
                missing_inputs.append(str(test_scores_path.relative_to(PROJECT_ROOT)))
            incomplete_experiments.append(
                f"{experiment_dir.name} (missing: {', '.join(missing_inputs)})"
            )

    if incomplete_experiments:
        raise FileNotFoundError(
            "Each experiment requires embeddings/embeddings.parquet or both score "
            "files. Incomplete experiments: " + "; ".join(incomplete_experiments)
        )

    return embedding_experiments, score_experiments


def run_step(step_name, script_path, arguments=()):
    command = [sys.executable, str(script_path), *arguments]
    # print(SEPARATOR)
    # print(step_name)
    # print(SEPARATOR)
    # print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print()


def main():
    embedding_experiments, score_experiments = experiment_input_modes()

    if embedding_experiments:
        print(
            "Generating scores from embeddings for: "
            + ", ".join(embedding_experiments)
        )
        for step_name, script_path in EMBEDDING_STEPS:
            run_step(
                step_name,
                script_path,
                ("--experiments", *embedding_experiments),
            )

    if score_experiments:
        print(
            "Using existing scores and skipping the embedding and score-generation "
            "steps for: " + ", ".join(score_experiments)
        )
        print()

    for step_name, script_path in PIPELINE_STEPS:
        run_step(step_name, script_path)


if __name__ == "__main__":
    main()
