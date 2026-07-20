"""
Create speaker-level enrolment embeddings for the dev and test splits.

For each experiment, this script loads utterance-level embeddings, selects the
enrolment utterances listed in data/shared, L2-normalizes each utterance
embedding, averages them per speaker, and saves one profile per speaker.

Inputs:
    data/shared/dev_enrolls.csv
    data/shared/test_enrolls.csv
    <experiment>/embeddings/embeddings.parquet

Outputs:
    <experiment>/embeddings/enrolment/dev_enroll_embeddings.parquet
    <experiment>/embeddings/enrolment/test_enroll_embeddings.parquet
"""

import argparse

import numpy as np
import pandas as pd

from tools.utils import (
    PROJECT_ROOT,
    enrolment_embeddings_path,
    iter_experiment_dirs,
    original_embeddings_path,
    SEPARATOR,
    SEPARATOR2,
)


SHARED_DIR = PROJECT_ROOT / "data" / "shared"
ENROLMENT_SPLITS = {
    "dev": SHARED_DIR / "dev_enrolls.csv",
    "test": SHARED_DIR / "test_enrolls.csv",
}




def average_enrolment_embeddings(embeddings_path, enroll_csv_path, output_parquet_path):
    df_emb = pd.read_parquet(embeddings_path)
    df_enrolls = pd.read_csv(enroll_csv_path)

    df_merged = df_enrolls.merge(
        df_emb.drop(columns=["speaker_id"], errors="ignore"),
        on="utterance_id",
        how="inner",
    )

    def normalize_and_mean(embs):
        stacked = np.stack(embs.values)
        normed = stacked / np.linalg.norm(stacked, axis=1, keepdims=True)
        return np.mean(normed, axis=0)

    df_avg_enrolls = (
        df_merged.groupby("speaker_id")["embedding"]
        .apply(normalize_and_mean)
        .reset_index()
    )
    utterances_per_profile = df_merged.groupby("speaker_id").size()

    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df_avg_enrolls.to_parquet(output_parquet_path, engine="pyarrow", index=False)

    return {
        "speaker_profiles": len(df_avg_enrolls),
        "enrolment_utterances": len(df_merged),
        "min_utterances_per_profile": utterances_per_profile.min(),
        "max_utterances_per_profile": utterances_per_profile.max(),
    }


def process_experiment(experiment_dir):
    embeddings_path = original_embeddings_path(experiment_dir)
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Missing required embeddings file: {embeddings_path}")

    split_summaries = {}
    for split, enroll_csv_path in ENROLMENT_SPLITS.items():
        if not enroll_csv_path.is_file():
            raise FileNotFoundError(f"Missing required enrolment file: {enroll_csv_path}")

        output_path = enrolment_embeddings_path(experiment_dir, split)
        split_summaries[split] = average_enrolment_embeddings(
            embeddings_path,
            enroll_csv_path,
            output_path,
        )

    return split_summaries


def print_summary(processed_experiments, split_summaries):
    print(f"Processing experiments: {', '.join(processed_experiments)}")
    for split in ENROLMENT_SPLITS:

        summaries = [summary[split] for summary in split_summaries]
        speaker_counts = {summary["speaker_profiles"] for summary in summaries}
        utterance_counts = {summary["enrolment_utterances"] for summary in summaries}
        min_utterances = min(
            summary["min_utterances_per_profile"] for summary in summaries
        )
        max_utterances = max(
            summary["max_utterances_per_profile"] for summary in summaries
        )

        if len(speaker_counts) == 1 and len(utterance_counts) == 1:
            speaker_text = f"{speaker_counts.pop()} speaker profiles"
            utterance_text = f"{utterance_counts.pop()} enrolment utterances"
        else:
            speaker_text = f"{min(speaker_counts)}-{max(speaker_counts)} speaker profiles"
            utterance_text = (
                f"{min(utterance_counts)}-{max(utterance_counts)} enrolment utterances"
            )
        print()
        print(
            f"{split}: {speaker_text} from {utterance_text} per experiment "
            f"(utterances per profile: min = {min_utterances}, max = {max_utterances})"
        )
    print()
    print("Aceraged embeddings are saved in each experiment's embeddings/enrolment/ directory.")


def main():
    parser = argparse.ArgumentParser(
        description="Create enrolment embeddings for one or more experiments."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        metavar="EXPERIMENT",
        help="Only process these experiment directory names.",
    )
    args = parser.parse_args()

    print()
    print(SEPARATOR)

    print("STEP 1. Creating speaker-level enrolment embeddings.")

    print(SEPARATOR)



    processed_experiments = []
    split_summaries = []
    for experiment_dir in iter_experiment_dirs(args.experiments):
        split_summaries.append(process_experiment(experiment_dir))
        processed_experiments.append(experiment_dir.name)

    print_summary(processed_experiments, split_summaries)
    print(SEPARATOR2)
    print()


if __name__ == "__main__":
    main()
