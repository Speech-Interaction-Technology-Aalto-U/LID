"""Precompute within-speaker cosine similarities between trial embeddings."""

from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("dev", "test")
SIMILARITY_FILENAME = "{split}_trial_embedding_similarities.csv"
SIMILARITY_COLUMNS = (
    "trial_spk",
    "trial_id_a",
    "trial_id_b",
    "cosine_similarity",
)


def similarity_path(experiment_dir, split):
    if split not in SPLITS:
        raise ValueError(f"Unknown data split: {split}")
    return Path(experiment_dir) / "scores" / SIMILARITY_FILENAME.format(split=split)


def _trial_metadata(scores_path):
    scores = pd.read_csv(
        scores_path,
        usecols=["trial_spk", "trial_id"],
        dtype={"trial_spk": "string", "trial_id": "string"},
    )
    if scores.empty:
        raise ValueError(f"{scores_path} contains no trial scores")
    metadata = scores.drop_duplicates("trial_id", keep="first")
    speaker_counts = scores.groupby("trial_id", sort=False)["trial_spk"].nunique()
    if (speaker_counts != 1).any():
        invalid = speaker_counts[speaker_counts != 1].index[:10].tolist()
        raise ValueError(
            "Each trial_id must map to one trial speaker before computing "
            f"embedding similarities. Invalid IDs: {invalid}"
        )
    return metadata


def compute_trial_embedding_similarities(
    embeddings_path,
    scores_path,
    output_path,
):
    """Write all within-speaker pairwise trial-embedding cosine similarities."""
    embeddings_path = Path(embeddings_path)
    scores_path = Path(scores_path)
    output_path = Path(output_path)
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Missing trial embeddings: {embeddings_path}")
    if not scores_path.is_file():
        raise FileNotFoundError(f"Missing trial scores: {scores_path}")

    metadata = _trial_metadata(scores_path)
    embeddings = pd.read_parquet(
        embeddings_path,
        columns=["utterance_id", "speaker_id", "embedding"],
    )
    embeddings["utterance_id"] = embeddings["utterance_id"].astype("string")
    embeddings["speaker_id"] = embeddings["speaker_id"].astype("string")
    duplicates = embeddings["utterance_id"].duplicated(keep=False)
    if duplicates.any():
        duplicate_ids = embeddings.loc[duplicates, "utterance_id"].head(10).tolist()
        raise ValueError(
            f"{embeddings_path} contains duplicate utterance IDs: {duplicate_ids}"
        )

    embedding_lookup = embeddings.set_index("utterance_id")
    missing_ids = metadata.loc[
        ~metadata["trial_id"].isin(embedding_lookup.index),
        "trial_id",
    ].tolist()
    if missing_ids:
        raise ValueError(
            f"{embeddings_path} is missing trial embeddings: {missing_ids[:10]}"
        )

    rows = []
    for trial_spk, speaker_trials in metadata.groupby("trial_spk", sort=False):
        trial_ids = speaker_trials["trial_id"].tolist()
        stored_speakers = embedding_lookup.loc[trial_ids, "speaker_id"].tolist()
        mismatched = [
            trial_id
            for trial_id, stored_speaker in zip(trial_ids, stored_speakers)
            if str(stored_speaker) != str(trial_spk)
        ]
        if mismatched:
            raise ValueError(
                "Embedding and score speaker identities disagree for trials: "
                f"{mismatched[:10]}"
            )

        matrix = np.stack(embedding_lookup.loc[trial_ids, "embedding"].to_numpy())
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError(
                f"Non-finite or inconsistent embeddings for speaker {trial_spk}"
            )
        norms = np.linalg.norm(matrix, axis=1)
        if (norms == 0).any():
            invalid = [
                trial_ids[index]
                for index in np.flatnonzero(norms == 0)[:10]
            ]
            raise ValueError(f"Zero-norm trial embeddings: {invalid}")
        normalized = matrix / norms[:, np.newaxis]
        similarities = np.clip(normalized @ normalized.T, -1.0, 1.0)
        indices_a, indices_b = np.triu_indices(len(trial_ids), k=1)
        rows.extend(
            {
                "trial_spk": trial_spk,
                "trial_id_a": trial_ids[index_a],
                "trial_id_b": trial_ids[index_b],
                "cosine_similarity": similarities[index_a, index_b],
            }
            for index_a, index_b in zip(indices_a, indices_b)
        )

    pairwise = pd.DataFrame(rows, columns=SIMILARITY_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairwise.to_csv(output_path, index=False)
    return {
        "path": output_path,
        "n_trials": int(len(metadata)),
        "n_speakers": int(metadata["trial_spk"].nunique()),
        "n_pairs": int(len(pairwise)),
    }


def precompute_experiment_embedding_similarities(experiment_dir):
    """Precompute development and test pairwise similarities for an experiment."""
    experiment_dir = Path(experiment_dir)
    embeddings_path = experiment_dir / "embeddings" / "embeddings.parquet"
    summaries = {}
    for split in SPLITS:
        summaries[split] = compute_trial_embedding_similarities(
            embeddings_path,
            experiment_dir / "scores" / f"{split}_scores.csv",
            similarity_path(experiment_dir, split),
        )
    return summaries


def load_trial_embedding_similarities(experiment_dir, split):
    """Load and validate precomputed within-speaker embedding similarities."""
    path = similarity_path(experiment_dir, split)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing precomputed trial-embedding similarities: {path}"
        )
    pairwise = pd.read_csv(
        path,
        dtype={
            "trial_spk": "string",
            "trial_id_a": "string",
            "trial_id_b": "string",
        },
    )
    missing = set(SIMILARITY_COLUMNS) - set(pairwise.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    pairwise["cosine_similarity"] = pd.to_numeric(
        pairwise["cosine_similarity"],
        errors="coerce",
    )
    similarities = pairwise["cosine_similarity"].to_numpy()
    if not np.isfinite(similarities).all() or (
        (similarities < -1.0) | (similarities > 1.0)
    ).any():
        raise ValueError(f"{path} contains invalid cosine similarities")
    duplicates = pairwise.duplicated(
        ["trial_spk", "trial_id_a", "trial_id_b"],
        keep=False,
    )
    if duplicates.any():
        examples = pairwise.loc[
            duplicates,
            ["trial_spk", "trial_id_a", "trial_id_b"],
        ].head(10)
        raise ValueError(
            f"{path} contains duplicate trial pairs: {examples.to_dict('records')}"
        )
    return pairwise
