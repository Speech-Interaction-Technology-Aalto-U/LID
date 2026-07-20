"""
This script creates cosine similarity scores for the dev and test trial embeddings compared to all enrolment utterances.

For each experiment, this script loads utterance-level embeddings and speaker-level enrolment profiles, selects the trial utterances listed in data/shared, L2-normalizes both sides, scores every enrolment profile against every valid trial utterance, and saves the scores in long CSV file.

Inputs:
    data/shared/dev_trials.csv
    data/shared/test_trials.csv
    <experiment>/embeddings/embeddings.parquet
    <experiment>/embeddings/enrolment/dev_enroll_embeddings.parquet
    <experiment>/embeddings/enrolment/test_enroll_embeddings.parquet

Outputs:
    <experiment>/scores/dev_scores.csv
    <experiment>/scores/test_scores.csv
"""

import argparse

import numpy as np
import pandas as pd

from tools.utils import (
    PROJECT_ROOT,
    enrolment_embeddings_path,
    iter_experiment_dirs,
    original_embeddings_path,
    scores_path,
    SEPARATOR,
    SEPARATOR2,
)

SHARED_DIR = PROJECT_ROOT / "data" / "shared"
SPLITS = {
    "dev": SHARED_DIR / "dev_trials.csv",
    "test": SHARED_DIR / "test_trials.csv",
}


def l2_normalize(matrix):
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def compute_scores(embeddings_path, enroll_embeddings_path, trials_csv_path, output_csv_path):
    df_emb = pd.read_parquet(embeddings_path)
    df_enroll = pd.read_parquet(enroll_embeddings_path)
    df_trials = pd.read_csv(trials_csv_path)

    emb_lookup = dict(zip(df_emb["utterance_id"], df_emb["embedding"]))
    valid_trials = df_trials[df_trials["trial_id"].isin(emb_lookup)].copy()

    enroll_spks = df_enroll["speaker_id"].values
    enroll_embs = l2_normalize(np.stack(df_enroll["embedding"].values))

    trial_ids = valid_trials["trial_id"].values
    trial_spks = valid_trials["trial_spk"].values
    trial_embs = l2_normalize(np.stack([emb_lookup[trial_id] for trial_id in trial_ids]))

    scores = np.dot(enroll_embs, trial_embs.T)
    df_scores = pd.DataFrame(scores, index=enroll_spks, columns=trial_ids)
    df_scores = df_scores.reset_index().melt(
        id_vars="index",
        var_name="trial_id",
        value_name="score",
    )
    df_scores = df_scores.rename(columns={"index": "enroll_spk"})
    df_scores["trial_spk"] = df_scores["trial_id"].map(dict(zip(trial_ids, trial_spks)))
    df_scores = df_scores[["enroll_spk", "trial_spk", "trial_id", "score"]]

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_scores.to_csv(output_csv_path, index=False)

    return {
        "scores": len(df_scores),
        "enrolment_profiles": len(enroll_spks),
        "trial_utterances": len(valid_trials),
        "missing_trial_utterances": len(df_trials) - len(valid_trials),
    }


def process_experiment(experiment_dir):
    embeddings_path = original_embeddings_path(experiment_dir)
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Missing required embeddings file: {embeddings_path}")

    split_summaries = {}
    for split, trials_csv_path in SPLITS.items():
        enroll_embeddings_path = enrolment_embeddings_path(experiment_dir, split)
        if not enroll_embeddings_path.is_file():
            raise FileNotFoundError(
                f"Missing required enrolment embeddings file: {enroll_embeddings_path}"
            )
        if not trials_csv_path.is_file():
            raise FileNotFoundError(f"Missing required trials file: {trials_csv_path}")

        output_path = scores_path(experiment_dir, split)
        split_summaries[split] = compute_scores(
            embeddings_path,
            enroll_embeddings_path,
            trials_csv_path,
            output_path,
        )

    return split_summaries


def count_text(values, label):
    if len(values) == 1:
        return f"{values.pop()} {label}"
    return f"{min(values)}-{max(values)} {label}"


def print_summary(split_summaries):

    for split in SPLITS:
        summaries = [summary[split] for summary in split_summaries]
        score_text = count_text({summary["scores"] for summary in summaries}, "scores")
        enrolment_text = count_text(
            {summary["enrolment_profiles"] for summary in summaries},
            "enrolment profiles",
        )
        trial_text = count_text(
            {summary["trial_utterances"] for summary in summaries},
            "trial utterances",
        )
        missing_trials = sum(
            summary["missing_trial_utterances"] for summary in summaries
        )

        print(
            f"{split}: {score_text} from {enrolment_text} and {trial_text} "
            "(per experiment)"
        )
        print()
        if missing_trials:
            print(
                f"  skipped {missing_trials} trial utterances without embeddings "
                "across all experiments"
            )
    print("The scores are saved in each experiment's scores/ directory.")


def main():
    parser = argparse.ArgumentParser(
        description="Create similarity scores for one or more experiments."
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
    print("STEP 2. Creating similarity scores.")
    print(SEPARATOR)
    print(
        "Matching all (dev/test) trial utterance embeddings against all (dev/test) enrolled speaker profiles."
    )
    print()

    split_summaries = []
    for experiment_dir in iter_experiment_dirs(args.experiments):
        split_summaries.append(process_experiment(experiment_dir))

    print_summary(split_summaries)
    print(SEPARATOR2)
    print()


if __name__ == "__main__":
    main()
