from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
SEPARATOR = "=" * 72
SEPARATOR2 = "-" * 72

EMBEDDINGS_DIRNAME = "embeddings"
ENROLMENT_DIRNAME = "enrolment"
SCORES_DIRNAME = "scores"
OUTPUTS_DIRNAME = "outputs"

EMBEDDINGS_FILE = "embeddings.parquet"
CALIBRATION_PARAMETERS_FILE = "calibration_parameters.json"
LID_FILE = "local_information_disclosures.csv"


def relative_path(path):
    return path.relative_to(PROJECT_ROOT)


def iter_experiment_dirs(experiment_names=None):
    if not EXPERIMENTS_DIR.is_dir():
        raise FileNotFoundError(f"Missing experiments directory: {EXPERIMENTS_DIR}")

    experiment_dirs = sorted(path for path in EXPERIMENTS_DIR.iterdir() if path.is_dir())
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiment directories found in: {EXPERIMENTS_DIR}")

    if experiment_names is not None:
        requested_names = set(experiment_names)
        available_names = {path.name for path in experiment_dirs}
        missing_names = sorted(requested_names - available_names)
        if missing_names:
            raise FileNotFoundError(
                f"Unknown experiment directories: {', '.join(missing_names)}"
            )
        experiment_dirs = [
            path for path in experiment_dirs if path.name in requested_names
        ]

    return experiment_dirs


def embeddings_dir(experiment_dir: Path) -> Path:
    return experiment_dir / EMBEDDINGS_DIRNAME


def enrolment_dir(experiment_dir: Path) -> Path:
    return embeddings_dir(experiment_dir) / ENROLMENT_DIRNAME


def scores_dir(experiment_dir: Path) -> Path:
    return experiment_dir / SCORES_DIRNAME


def outputs_dir(experiment_dir: Path) -> Path:
    return experiment_dir / OUTPUTS_DIRNAME


def original_embeddings_path(experiment_dir: Path) -> Path:
    return embeddings_dir(experiment_dir) / EMBEDDINGS_FILE


def enrolment_embeddings_path(experiment_dir: Path, split: str) -> Path:
    return enrolment_dir(experiment_dir) / f"{split}_enroll_embeddings.parquet"


def scores_path(experiment_dir: Path, split: str) -> Path:
    return scores_dir(experiment_dir) / f"{split}_scores.csv"


def calibration_parameters_path(experiment_dir: Path) -> Path:
    return outputs_dir(experiment_dir) / CALIBRATION_PARAMETERS_FILE


def lid_path(experiment_dir: Path) -> Path:
    return outputs_dir(experiment_dir) / LID_FILE
