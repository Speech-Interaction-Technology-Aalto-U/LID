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
    _similarity_statistics,
    aggregate_logits,
    analyse_calibrated_aggregations,
    build_calibration_dataset,
    fit_method_parameter,
    load_development_scores,
)


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


class LongitudinalCalibrationTests(unittest.TestCase):

    def test_count_and_similarity_endpoints_recover_sum_and_average(self):
        sums = np.array([[6.0, 3.0], [2.0, 8.0]])
        counts = np.array([3.0, 4.0])
        dataset = CalibrationDataset(
            sum_llr=sums,
            average_llr=sums / counts[:, None],
            counts=counts,
            redundancy=counts - 1.0,
            mean_positive_similarity=np.ones(2),
            target_indices=np.array([0, 1]),
            speaker_ids=("A", "B"),
        )

        count_sum, _ = aggregate_logits(dataset, "count_adjusted", 0.0)
        count_average, _ = aggregate_logits(dataset, "count_adjusted", 1.0)
        similarity_average, _ = aggregate_logits(
            dataset,
            "similarity_adjusted",
            1.0,
        )

        np.testing.assert_allclose(count_sum, sums)
        np.testing.assert_allclose(count_average, dataset.average_llr)
        np.testing.assert_allclose(similarity_average, dataset.average_llr)

    def test_identical_observations_have_full_redundancy(self):
        matrix = np.repeat([[2.0, -1.0, 0.5]], repeats=4, axis=0)
        redundancy, mean_similarity = _similarity_statistics(matrix)

        self.assertAlmostEqual(redundancy, 3.0)
        self.assertAlmostEqual(mean_similarity, 1.0)

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

            report = (output_dir / "interactive_longitudinal.html").read_text()
            self.assertIn("Slider temperature T", report)
            self.assertIn('"automatic_test_based_model_selection":false', report)
            self.assertNotIn("https://", report)


if __name__ == "__main__":
    unittest.main()
