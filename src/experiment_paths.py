from pathlib import Path


EMBEDDINGS_DIRNAME = "embeddings"
ENROLMENT_DIRNAME = "enrolment"
SCORES_DIRNAME = "scores"
OUTPUTS_DIRNAME = "outputs"

EMBEDDINGS_FILE = "embeddings.parquet"
CALIBRATION_PARAMETERS_FILE = "calibration_parameters.json"
LID_FILE = "local_information_disclosures.csv"


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
