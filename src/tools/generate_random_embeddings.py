"""
Create a random-embedding baseline from an existing experiment schema.

The generated random embeddings use the utterance metadata and embedding
dimension from a reference experiment, but only for utterances listed in the
shared dev/test enrolment and trial files.

Example:
    uv run src/tools/generate_random_embeddings.py B3
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tools.utils import (
    EXPERIMENTS_DIR,
    PROJECT_ROOT,
    original_embeddings_path,
    relative_path,
    SEPARATOR,
    SEPARATOR2,
)

SHARED_DIR = PROJECT_ROOT / "data" / "shared"
DEFAULT_REFERENCE_EXPERIMENT = "B3"
DEFAULT_OUTPUT_EXPERIMENT = "random"
SHARED_ID_FILES = (
    ("dev enrolments", SHARED_DIR / "dev_enrolls.csv", "utterance_id"),
    ("dev trials", SHARED_DIR / "dev_trials.csv", "utterance_id"),
    ("test enrolments", SHARED_DIR / "test_enrolls.csv", "utterance_id"),
    ("test trials", SHARED_DIR / "test_trials.csv", "utterance_id"),
)


def load_shared_utterance_ids(shared_dir):
    ids = []
    seen = set()

    for label, csv_path, id_column in SHARED_ID_FILES:
        csv_path = shared_dir / csv_path.name
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing required {label} file: {csv_path}")

        df = pd.read_csv(csv_path)
        if id_column not in df.columns:
            raise ValueError(f"Expected column {id_column!r} in {csv_path}")

        for utterance_id in df[id_column]:
            if utterance_id not in seen:
                seen.add(utterance_id)
                ids.append(utterance_id)

    return ids


def generate_random_embeddings(
    reference_experiment=DEFAULT_REFERENCE_EXPERIMENT,
    output_experiment=DEFAULT_OUTPUT_EXPERIMENT,
    shared_dir=SHARED_DIR,
    seed=None,
):
    reference_experiment_dir = EXPERIMENTS_DIR / reference_experiment
    output_experiment_dir = EXPERIMENTS_DIR / output_experiment
    reference_parquet_path = original_embeddings_path(reference_experiment_dir)
    output_parquet_path = original_embeddings_path(output_experiment_dir)
    shared_dir = Path(shared_dir)

    if not reference_parquet_path.is_file():
        raise FileNotFoundError(
            f"Missing reference embeddings file: {reference_parquet_path}"
        )

    df_reference = pd.read_parquet(reference_parquet_path)

    required_columns = {"utterance_id", "speaker_id", "embedding", "source_file"}
    missing_columns = required_columns - set(df_reference.columns)
    if missing_columns:
        raise ValueError(
            f"Reference parquet is missing required columns: {sorted(missing_columns)}"
        )

    shared_ids = load_shared_utterance_ids(shared_dir)
    df = (
        pd.DataFrame({"utterance_id": shared_ids})
        .merge(
            df_reference.drop(columns=["embedding"]),
            on="utterance_id",
            how="left",
            validate="one_to_one",
        )
    )

    missing_reference_ids = df[df["speaker_id"].isna()]["utterance_id"].tolist()
    if missing_reference_ids:
        sample = ", ".join(missing_reference_ids[:5])
        raise ValueError(
            f"{len(missing_reference_ids)} utterance IDs from {relative_path(shared_dir)} "
            f"were not found in {relative_path(reference_parquet_path)}. "
            f"First missing IDs: {sample}"
        )

    reference_embedding = df_reference["embedding"].iloc[0]
    embedding_dim = len(reference_embedding)
    rng = np.random.default_rng(seed)

    df["embedding"] = list(
        rng.uniform(-100, 100, size=(len(df), embedding_dim)).astype(np.float32)
    )
    df = df[df_reference.columns]

    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet_path, index=False)

    return {
        "reference_experiment": reference_experiment,
        "output_experiment": output_experiment,
        "embeddings": len(df),
        "embedding_dim": embedding_dim,
        "output_path": output_parquet_path,
        "seed": seed,
    }


def print_summary(summary):
    print(
        f"Created {summary['embeddings']} random "
        f"{summary['embedding_dim']}-dimensional embeddings."
    )
    print(
        f"Reference experiment: {summary['reference_experiment']} -> "
        f"output experiment: {summary['output_experiment']}"
    )
    if summary["seed"] is not None:
        print(f"Seed: {summary['seed']}")
    print()
    print(f"Saved to {relative_path(summary['output_path'])}.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate random embeddings matching a reference experiment's schema "
            "and embedding dimension."
        )
    )
    parser.add_argument(
        "reference_experiment",
        nargs="?",
        default=DEFAULT_REFERENCE_EXPERIMENT,
        help=(
            "Experiment under data/experiments used as the schema/dimension "
            f"reference. Default: {DEFAULT_REFERENCE_EXPERIMENT}."
        ),
    )
    parser.add_argument(
        "--output-experiment",
        default=DEFAULT_OUTPUT_EXPERIMENT,
        help=(
            "Experiment folder where random embeddings are written. "
            f"Default: {DEFAULT_OUTPUT_EXPERIMENT}."
        ),
    )
    parser.add_argument(
        "--shared-dir",
        type=Path,
        default=SHARED_DIR,
        help="Directory containing dev/test enrolment and trial CSV files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print()
    print(SEPARATOR)
    print("Generating random baseline embeddings.")
    print(SEPARATOR)
    print(
        "Using shared dev/test utterance IDs and matching the reference "
        "experiment's embedding dimension."
    )
    print()

    summary = generate_random_embeddings(
        reference_experiment=args.reference_experiment,
        output_experiment=args.output_experiment,
        shared_dir=args.shared_dir,
        seed=args.seed,
    )
    print_summary(summary)
    print(SEPARATOR2)
    print()
