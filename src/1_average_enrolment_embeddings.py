import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from experiment_paths import enrolment_embeddings_path, original_embeddings_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
SHARED_DIR = PROJECT_ROOT / "data" / "shared"
ENROLLMENT_SPLITS = {
    "dev": SHARED_DIR / "dev_enrolls.csv",
    "test": SHARED_DIR / "test_enrolls.csv",
}
SEPARATOR = "-" * 72


def relative_path(path):
    return path.relative_to(PROJECT_ROOT)


def average_enrollment_embeddings(embeddings_path, enroll_csv_path, output_parquet_path):
    df_emb = pd.read_parquet(embeddings_path)
    df_enrolls = pd.read_csv(enroll_csv_path)
    
    # 1. Merge the enrollment list with the raw embeddings to extract the right vectors
    df_merged = df_enrolls.merge(
        df_emb.drop(columns=["speaker_id"], errors="ignore"), 
        on="utterance_id", 
        how="inner"
    )

    # Function to L2 normalize utterance vectors, then average them per speaker
    def normalize_and_mean(embs):
        stacked = np.stack(embs.values)
        # Normalize each individual utterance embedding
        normed = stacked / np.linalg.norm(stacked, axis=1, keepdims=True)
        # Return the mean of these normalized embeddings
        return np.mean(normed, axis=0)

    # 2. Group by speaker_id and apply the averaging function
    df_avg_enrolls = (
        df_merged.groupby("speaker_id")["embedding"]
        .apply(normalize_and_mean)
        .reset_index()
    )
    utterances_per_profile = df_merged.groupby("speaker_id").size()

    # 3. Save the resulting speaker profiles
    Path(output_parquet_path).parent.mkdir(parents=True, exist_ok=True)
    df_avg_enrolls.to_parquet(output_parquet_path, engine="pyarrow", index=False)

    print(
        f"{len(df_avg_enrolls)} speaker profiles created by averaging "
        f"{len(df_merged)} enrolment utterances "
        f"(utterances per profile: min = {utterances_per_profile.min()}, "
        f"max = {utterances_per_profile.max()})"
    )
    print(f"averaged embeddings saved in {relative_path(output_parquet_path)}")


def process_experiment(experiment_dir):
    embeddings_path = original_embeddings_path(experiment_dir)
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Missing required embeddings file: {embeddings_path}")

    print(SEPARATOR)
    print(f"Experiment: {experiment_dir.name}")
    print(SEPARATOR)
    for split, enroll_csv_path in ENROLLMENT_SPLITS.items():
        if not enroll_csv_path.is_file():
            raise FileNotFoundError(f"Missing required enrollment file: {enroll_csv_path}")

        output_path = enrolment_embeddings_path(experiment_dir, split)
        print(f"{split}:")
        average_enrollment_embeddings(embeddings_path, enroll_csv_path, output_path)
        print()


def iter_experiment_dirs():
    if not EXPERIMENTS_DIR.is_dir():
        raise FileNotFoundError(f"Missing experiments directory: {EXPERIMENTS_DIR}")

    experiment_dirs = sorted(path for path in EXPERIMENTS_DIR.iterdir() if path.is_dir())
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiment directories found in: {EXPERIMENTS_DIR}")
    return experiment_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average enrollment embeddings per speaker.")
    parser.add_argument(
        "experiment",
        nargs="?",
        help="Experiment name under data/experiments. If omitted, all experiments are processed.",
    )
    
    args = parser.parse_args()

    print("Creating speaker-level enrolment embeddings.")
    print("For each experiment, dev and test enrolment utterances are matched to")
    print("embeddings/embeddings.parquet, L2-normalized, averaged per speaker, and saved")
    print("under embeddings/enrolment/.\n")

    if args.experiment:
        process_experiment(EXPERIMENTS_DIR / args.experiment)
    else:
        for experiment_dir in iter_experiment_dirs():
            process_experiment(experiment_dir)
