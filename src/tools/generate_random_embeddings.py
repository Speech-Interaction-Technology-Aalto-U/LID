import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment_paths import original_embeddings_path

SHARED_DIR = PROJECT_ROOT / "data" / "shared"
DEFAULT_REFERENCE_PARQUET = original_embeddings_path(
    PROJECT_ROOT / "data" / "experiments" / "B3_ECAPA"
)
DEFAULT_OUTPUT_PARQUET = original_embeddings_path(
    PROJECT_ROOT / "data" / "experiments" / "random"
)
SHARED_ID_FILES = (
    ("dev enrolments", SHARED_DIR / "dev_enrolls.csv", "utterance_id"),
    ("dev trials", SHARED_DIR / "dev_trials.csv", "trial_id"),
    ("test enrolments", SHARED_DIR / "test_enrolls.csv", "utterance_id"),
    ("test trials", SHARED_DIR / "test_trials.csv", "trial_id"),
)


def relative_path(path):
    return path.relative_to(PROJECT_ROOT)


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
    reference_parquet_path=DEFAULT_REFERENCE_PARQUET,
    output_parquet_path=DEFAULT_OUTPUT_PARQUET,
    shared_dir=SHARED_DIR,
    seed=None,
):
    reference_parquet_path = Path(reference_parquet_path)
    output_parquet_path = Path(output_parquet_path)
    shared_dir = Path(shared_dir)

    print(f"Loading reference from {relative_path(reference_parquet_path)}")
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

    print(f"Generating {len(df)} random {embedding_dim}-dimensional embeddings...")
    df["embedding"] = list(
        rng.uniform(-100, 100, size=(len(df), embedding_dim)).astype(np.float32)
    )
    df = df[df_reference.columns]

    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {relative_path(output_parquet_path)}")
    df.to_parquet(output_parquet_path, index=False)
    print("Done!")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate random embeddings matching the B3_ECAPA parquet schema for "
            "utterances listed in data/shared."
        )
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_PARQUET,
        help="Reference embeddings parquet used for schema and utterance metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PARQUET,
        help="Output parquet path.",
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
    generate_random_embeddings(args.reference, args.output, args.shared_dir, args.seed)
