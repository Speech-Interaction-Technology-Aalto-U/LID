import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from long_leakage.analysis import build_longitudinal_evidence
from long_leakage.calibration import (
    CalibrationDataset,
    _multiclass_metrics,
    aggregate_logits,
    analyse_calibrated_aggregations,
    build_calibration_dataset,
    fit_method_parameter,
    fit_method_parameters,
    load_development_scores,
)
from long_leakage.similarity import compute_trial_embedding_similarities


def longitudinal_scores(scale=1.0):
    vectors = [
        ("a-1", "A", [2.2, 1.0, -0.2]),
        ("a-2", "A", [1.5, 1.7, -0.5]),
        ("a-3", "A", [2.0, 0.8, 0.0]),
        ("b-1", "B", [0.8, 2.4, 0.2]),
        ("b-2", "B", [1.2, 1.9, 0.0]),
        ("b-3", "B", [0.4, 2.1, 0.7]),
        ("c-1", "C", [0.1, 1.0, 2.0]),
        ("c-2", "C", [0.2, 1.5, 1.7]),
        ("c-3", "C", [-0.2, 0.8, 2.1]),
    ]
    rows = []
    for trial_id, trial_spk, llrs in vectors:
        for enroll_spk, llr in zip(("A", "B", "C"), llrs):
            rows.append(
                {
                    "enroll_spk": enroll_spk,
                    "trial_spk": trial_spk,
                    "trial_id": trial_id,
                    "llr": scale * llr,
                }
            )
    return pd.DataFrame(rows)


def longitudinal_embeddings():
    vectors = {
        "a-1": ("A", [1.0, 0.0, 0.0]),
        "a-2": ("A", [0.9, 0.1, 0.0]),
        "a-3": ("A", [0.8, 0.2, 0.1]),
        "b-1": ("B", [0.0, 1.0, 0.0]),
        "b-2": ("B", [0.1, 0.9, 0.0]),
        "b-3": ("B", [0.2, 0.8, 0.1]),
        "c-1": ("C", [0.0, 0.0, 1.0]),
        "c-2": ("C", [0.0, 0.1, 0.9]),
        "c-3": ("C", [0.1, 0.2, 0.8]),
    }
    return pd.DataFrame(
        [
            {
                "utterance_id": utterance_id,
                "speaker_id": speaker_id,
                "embedding": np.asarray(embedding, dtype=float),
            }
            for utterance_id, (speaker_id, embedding) in vectors.items()
        ]
    )


class LongitudinalCalibrationTests(unittest.TestCase):

    def test_count_and_similarity_endpoints_recover_sum_and_average(self):
        sums = np.array([[6.0, 3.0], [2.0, 8.0]])
        counts = np.array([3.0, 4.0])
        dataset = CalibrationDataset(
            sum_llr=sums,
            average_llr=sums / counts[:, None],
            counts=counts,
            embedding_redundancy=counts - 1.0,
            mean_embedding_cosine_similarity=np.ones(2),
            mean_positive_embedding_similarity=np.ones(2),
            target_indices=np.array([0, 1]),
            speaker_ids=("A", "B"),
        )

        count_sum, _ = aggregate_logits(
            dataset,
            "count_adjusted",
            {"temperature": 1.0, "rho": 0.0},
        )
        count_average, _ = aggregate_logits(
            dataset,
            "count_adjusted",
            {"temperature": 1.0, "rho": 1.0},
        )
        similarity_average, _ = aggregate_logits(
            dataset,
            "similarity_adjusted",
            {"temperature": 1.0, "rho": 1.0},
        )
        count_global, _ = aggregate_logits(
            dataset,
            "count_adjusted",
            {"temperature": 2.0, "rho": 0.0},
        )

        np.testing.assert_allclose(count_sum, sums)
        np.testing.assert_allclose(count_average, dataset.average_llr)
        np.testing.assert_allclose(similarity_average, dataset.average_llr)
        np.testing.assert_allclose(count_global, sums / 2.0)

    def test_precomputes_within_speaker_embedding_cosines(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            scores_path = directory / "scores.csv"
            embeddings_path = directory / "embeddings.parquet"
            output_path = directory / "similarities.csv"
            pd.DataFrame(
                {
                    "trial_spk": ["A", "A", "B"],
                    "trial_id": ["a-1", "a-2", "b-1"],
                }
            ).to_csv(scores_path, index=False)
            pd.DataFrame(
                {
                    "utterance_id": ["a-1", "a-2", "b-1"],
                    "speaker_id": ["A", "A", "B"],
                    "embedding": [
                        np.asarray([1.0, 0.0]),
                        np.asarray([0.6, 0.8]),
                        np.asarray([0.0, 1.0]),
                    ],
                }
            ).to_parquet(embeddings_path, index=False)

            summary = compute_trial_embedding_similarities(
                embeddings_path,
                scores_path,
                output_path,
            )
            pairwise = pd.read_csv(output_path)

            self.assertEqual(summary["n_pairs"], 1)
            self.assertEqual(pairwise.loc[0, "trial_spk"], "A")
            self.assertAlmostEqual(pairwise.loc[0, "cosine_similarity"], 0.6)

    def test_fitted_temperature_minimizes_development_log_loss(self):
        dataset = build_calibration_dataset(
            build_longitudinal_evidence(longitudinal_scores(scale=4.0))
        )
        fitted = fit_method_parameter(dataset, "temperature_scaled")
        fitted_logits, _ = aggregate_logits(
            dataset,
            "temperature_scaled",
            fitted,
        )
        sum_logits, _ = aggregate_logits(dataset, "sum_llr")

        fitted_nll = _multiclass_metrics(
            fitted_logits,
            dataset.target_indices,
        )["multiclass_nll_nats"]
        summed_nll = _multiclass_metrics(
            sum_logits,
            dataset.target_indices,
        )["multiclass_nll_nats"]
        self.assertGreater(fitted, 0.0)
        self.assertLessEqual(fitted_nll, summed_nll + 1e-10)

    def test_joint_fit_profiles_temperature_to_find_interior_rho(self):
        margins = np.asarray([2.0, 2.0, -1.0, 6.0, 6.0, -3.0])
        sums = np.column_stack((margins, np.zeros(len(margins))))
        counts = np.asarray([2.0, 2.0, 2.0, 10.0, 10.0, 10.0])
        dataset = CalibrationDataset(
            sum_llr=sums,
            average_llr=sums / counts[:, None],
            counts=counts,
            embedding_redundancy=np.zeros(len(sums)),
            mean_embedding_cosine_similarity=np.zeros(len(sums)),
            mean_positive_embedding_similarity=np.zeros(len(sums)),
            target_indices=np.zeros(len(sums), dtype=int),
            speaker_ids=tuple(str(index) for index in range(len(sums))),
        )

        parameters = fit_method_parameters(dataset, "count_adjusted")

        self.assertAlmostEqual(parameters["rho"], 1.0 / 3.0, places=5)
        self.assertGreater(parameters["temperature"], 0.0)

    def test_nll_and_brier_temperature_fits_optimize_different_scores(self):
        sums = np.asarray(
            [
                [0.604, -0.374, 3.564],
                [1.412, -1.516, -0.011],
                [-1.336, 0.318, -1.445],
                [2.518, 0.504, 3.375],
                [0.678, 3.094, -3.198],
                [4.826, -4.104, 1.860],
                [-1.207, -1.886, -1.406],
                [-1.440, 0.314, -0.236],
            ]
        )
        targets = np.asarray([0, 1, 2, 0, 1, 2, 0, 1])
        dataset = CalibrationDataset(
            sum_llr=sums,
            average_llr=sums,
            counts=np.ones(len(sums)),
            embedding_redundancy=np.zeros(len(sums)),
            mean_embedding_cosine_similarity=np.zeros(len(sums)),
            mean_positive_embedding_similarity=np.zeros(len(sums)),
            target_indices=targets,
            speaker_ids=tuple(str(index) for index in range(len(sums))),
        )
        nll_parameters = fit_method_parameters(
            dataset,
            "temperature_scaled",
        )
        brier_parameters = fit_method_parameters(
            dataset,
            "temperature_scaled_brier",
        )
        nll_logits, _ = aggregate_logits(
            dataset,
            "temperature_scaled",
            nll_parameters,
        )
        brier_logits, _ = aggregate_logits(
            dataset,
            "temperature_scaled_brier",
            brier_parameters,
        )
        nll_metrics = _multiclass_metrics(nll_logits, targets)
        brier_metrics = _multiclass_metrics(brier_logits, targets)

        self.assertNotAlmostEqual(
            nll_parameters["temperature"],
            brier_parameters["temperature"],
            places=2,
        )
        self.assertLessEqual(
            nll_metrics["multiclass_nll_nats"],
            brier_metrics["multiclass_nll_nats"] + 1e-8,
        )
        self.assertLessEqual(
            brier_metrics["multiclass_brier"],
            nll_metrics["multiclass_brier"] + 1e-8,
        )

    def test_loads_development_llrs_from_existing_calibration(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir) / "experiment"
            (experiment_dir / "scores").mkdir(parents=True)
            (experiment_dir / "outputs").mkdir()
            scores = longitudinal_scores().drop(columns="llr")
            scores["z_score"] = np.linspace(-1.0, 1.0, len(scores))
            scores.to_csv(experiment_dir / "scores" / "dev_scores.csv", index=False)
            (experiment_dir / "outputs" / "calibration_parameters.json").write_text(
                json.dumps({"w": 2.5, "b": -0.75})
            )

            loaded = load_development_scores(experiment_dir)

            np.testing.assert_allclose(
                loaded["llr"],
                2.5 * scores["z_score"] - 0.75,
            )

    def test_writes_calibrated_methods_and_offline_interactive_report(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir) / "toy"
            scores_dir = experiment_dir / "scores"
            scores_dir.mkdir(parents=True)
            longitudinal_scores(scale=1.2).to_csv(
                scores_dir / "dev_scores.csv",
                index=False,
            )
            longitudinal_scores(scale=1.0).to_csv(
                scores_dir / "test_scores.csv",
                index=False,
            )
            embeddings_dir = experiment_dir / "embeddings"
            embeddings_dir.mkdir()
            longitudinal_embeddings().to_parquet(
                embeddings_dir / "embeddings.parquet",
                index=False,
            )

            summary = analyse_calibrated_aggregations(experiment_dir, dpi=35)
            output_dir = experiment_dir / "long"

            self.assertGreater(summary["fitted_temperature"], 0.0)
            for filename in (
                "calibration.json",
                "calibration_diagnostics.csv",
                "calibration_folds.csv",
                "temperature_sweep.csv",
                "method_comparison.csv",
                "speaker_redundancy.csv",
                "interactive_longitudinal.html",
            ):
                self.assertTrue((output_dir / filename).is_file())

            for method in (
                "temperature_scaled",
                "temperature_scaled_brier",
                "count_adjusted",
                "similarity_adjusted",
            ):
                scores = pd.read_csv(output_dir / method / "scores.csv")
                np.testing.assert_allclose(
                    scores.groupby("trial_spk")["p"].sum(),
                    1.0,
                )
                calibration = json.loads(
                    (output_dir / method / "summary.json").read_text()
                )["calibration"]
                self.assertEqual(
                    calibration["full_development_metrics"]["n_speakers"],
                    3,
                )

            for split in ("dev", "test"):
                similarity_file = (
                    scores_dir
                    / f"{split}_trial_embedding_similarities.csv"
                )
                self.assertTrue(similarity_file.is_file())
                self.assertEqual(len(pd.read_csv(similarity_file)), 9)

            report = (output_dir / "interactive_longitudinal.html").read_text()
            self.assertIn("Global temperature", report)
            self.assertIn("Use Brier fit", report)
            self.assertIn("Embedding-adjusted", report)
            self.assertIn("LOO expectation", report)
            self.assertIn('id="global-histogram"', report)
            self.assertIn('"automatic_test_based_model_selection":false', report)
            self.assertNotIn("https://", report)


if __name__ == "__main__":
    unittest.main()
