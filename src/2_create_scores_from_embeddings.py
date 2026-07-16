import argparse
import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
SHARED_DIR = PROJECT_ROOT / "data" / "shared"
SPLITS = {
    "dev": SHARED_DIR / "dev_trials.csv",
    "test": SHARED_DIR / "test_trials.csv",
}
SEPARATOR = "-" * 72


def relative_path(path):
    return path.relative_to(PROJECT_ROOT)


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

    print(
        f"{len(df_scores)} scores created from {len(enroll_spks)} enrolment profiles "
        f"and {len(valid_trials)} trial utterances"
    )
    print(f"scores saved in {relative_path(output_csv_path)}")


def process_experiment(experiment_dir):
    embeddings_path = experiment_dir / "embeddings.parquet"
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Missing required embeddings file: {embeddings_path}")

    print(SEPARATOR)
    print(f"Experiment: {experiment_dir.name}")
    print(SEPARATOR)
    for split, trials_csv_path in SPLITS.items():
        enroll_embeddings_path = experiment_dir / f"{split}_enroll_embeddings.parquet"
        if not enroll_embeddings_path.is_file():
            raise FileNotFoundError(
                f"Missing required enrolment embeddings file: {enroll_embeddings_path}"
            )
        if not trials_csv_path.is_file():
            raise FileNotFoundError(f"Missing required trials file: {trials_csv_path}")

        output_path = experiment_dir / f"{split}_scores.csv"
        print(f"{split}:")
        compute_scores(embeddings_path, enroll_embeddings_path, trials_csv_path, output_path)
        print()


def iter_experiment_dirs():
    if not EXPERIMENTS_DIR.is_dir():
        raise FileNotFoundError(f"Missing experiments directory: {EXPERIMENTS_DIR}")

    experiment_dirs = sorted(path for path in EXPERIMENTS_DIR.iterdir() if path.is_dir())
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiment directories found in: {EXPERIMENTS_DIR}")
    return experiment_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create cosine similarity scores.")
    parser.add_argument(
        "experiment",
        nargs="?",
        help="Experiment name under data/experiments. If omitted, all experiments are processed.",
    )

    args = parser.parse_args()

    print("Creating trial scores from speaker-level enrolment embeddings.")
    print("For each experiment, dev and test trial utterances are matched to")
    print("embeddings.parquet, cosine-scored against each enrolment profile, and saved.\n")

    if args.experiment:
        process_experiment(EXPERIMENTS_DIR / args.experiment)
    else:
        for experiment_dir in iter_experiment_dirs():
            process_experiment(experiment_dir)
