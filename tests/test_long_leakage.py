import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from long_leakage.analysis import (
    _mated_marker_coordinates,
    aggregate_lid_metrics,
    analyse_experiment,
    build_longitudinal_evidence,
    create_longitudinal_summary,
    load_trial_scores,
)


def toy_scores():
    trial_vectors = [
        ("a-1", "A", [2.0, 0.0, -1.0]),
        ("b-1", "B", [0.0, 3.0, 1.0]),
        ("a-2", "A", [1.0, 2.0, -1.0]),
    ]
    rows = []
    for trial_id, trial_spk, llrs in trial_vectors:
        for enroll_spk, llr in zip(("A", "B", "C"), llrs):
            rows.append(
                {
                    "enroll_spk": enroll_spk,
                    "trial_spk": trial_spk,
                    "trial_id": trial_id,
                    "llr": llr,
                }
            )
    return pd.DataFrame(rows)


class LongitudinalEvidenceTests(unittest.TestCase):
    def test_aggregates_unequal_trial_counts_and_normalizes_probabilities(self):
        evidence = build_longitudinal_evidence(toy_scores())

        self.assertEqual(evidence.enroll_speakers, ("A", "B", "C"))
        self.assertEqual(evidence.speaker_ids, ("A", "B"))
        self.assertEqual(evidence.trial_ids, ("a-1", "a-2", "b-1"))
        np.testing.assert_array_equal(evidence.speaker_trial_counts, [2, 1])
        np.testing.assert_allclose(
            evidence.sum_llr,
            [[3.0, 2.0, -2.0], [0.0, 3.0, 1.0]],
        )
        np.testing.assert_allclose(
            evidence.avg_llr,
            [[1.5, 1.0, -1.0], [0.0, 3.0, 1.0]],
        )
        np.testing.assert_allclose(evidence.raw_p.sum(axis=1), 1.0)
        np.testing.assert_allclose(evidence.sum_p.sum(axis=1), 1.0)
        np.testing.assert_allclose(evidence.avg_p.sum(axis=1), 1.0)

        scores = evidence.scores_dataframe()
        self.assertEqual(
            scores.columns.tolist(),
            ["enroll_spk", "trial_spk", "sum_llr", "avg_llr"],
        )

    def test_extracts_mated_probabilities_and_lid_at_each_stage(self):
        evidence = build_longitudinal_evidence(toy_scores())
        mated = evidence.mated_dataframe()

        self.assertEqual(
            mated.groupby("stage", sort=False).size().to_dict(),
            {"before": 3, "summed": 2, "averaged": 2},
        )
        expected_lid = np.log2(evidence.n_enrolments * mated["p"].to_numpy())
        np.testing.assert_allclose(mated["LID"], expected_lid)

    def test_centers_mated_markers_in_variable_height_speaker_rows(self):
        evidence = build_longitudinal_evidence(toy_scores())
        x_values, y_values = _mated_marker_coordinates(evidence)

        np.testing.assert_allclose(x_values, [0.5, 1.5])
        np.testing.assert_allclose(y_values, [1.0, 2.5])

    def test_orders_trial_speakers_to_match_enrolment_columns(self):
        scores = toy_scores()
        b_rows = scores[scores["trial_spk"] == "B"]
        a_rows = scores[scores["trial_spk"] == "A"]
        evidence = build_longitudinal_evidence(
            pd.concat((b_rows, a_rows), ignore_index=True)
        )

        self.assertEqual(evidence.enroll_speakers[:2], ("A", "B"))
        self.assertEqual(evidence.speaker_ids, ("A", "B"))
        self.assertEqual(evidence.trial_ids, ("a-1", "a-2", "b-1"))

    def test_computes_pipeline_compatible_averaged_lid_metrics(self):
        metrics = aggregate_lid_metrics(
            "toy",
            np.array([-2.0, 1.0, 3.0]),
            n_enrolments=3,
        )

        self.assertEqual(metrics["n_trials"], 3)
        self.assertAlmostEqual(metrics["ALID"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["PDR"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["NDR"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["LID+"], 2.0)
        self.assertAlmostEqual(metrics["LID-"], -2.0)
        self.assertAlmostEqual(metrics["LID_max"], 3.0)

    def test_rejects_incomplete_trial_matrix(self):
        incomplete = toy_scores().drop(index=0)
        with tempfile.TemporaryDirectory() as temporary_dir:
            scores_path = Path(temporary_dir) / "test_scores.csv"
            incomplete.to_csv(scores_path, index=False)
            with self.assertRaisesRegex(ValueError, "same enrolment set"):
                load_trial_scores(scores_path)

    def test_creates_requested_csvs_and_plots(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir) / "toy"
            scores_dir = experiment_dir / "scores"
            scores_dir.mkdir(parents=True)
            toy_scores().to_csv(scores_dir / "test_scores.csv", index=False)

            summary = analyse_experiment(experiment_dir, dpi=40)
            output_dir = experiment_dir / "long"

            self.assertEqual(summary["n_trials"], 3)
            self.assertEqual(summary["n_trial_speakers"], 2)
            self.assertEqual(
                pd.read_csv(output_dir / "scores.csv").columns.tolist(),
                ["enroll_spk", "trial_spk", "sum_llr", "avg_llr"],
            )
            probabilities = pd.read_csv(output_dir / "probabilities.csv")
            for probability_column in ("sum_p", "avg_p"):
                row_sums = probabilities.groupby("trial_spk")[probability_column].sum()
                np.testing.assert_allclose(row_sums, 1.0)

            expected_files = {
                "scores.csv",
                "probabilities.csv",
                "mated_probabilities.csv",
                "summary.json",
                "llr_heatmaps.png",
                "llr_heatmaps.pdf",
                "probability_heatmaps.png",
                "probability_heatmaps.pdf",
                "mated_probability_comparison.png",
                "mated_probability_comparison.pdf",
                "mated_lid_comparison.png",
                "mated_lid_comparison.pdf",
                "sum_llr",
                "average_llr",
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                expected_files,
            )
            for plot_path in summary["plot_paths"]:
                self.assertGreater(plot_path.stat().st_size, 0)

            method_files = {
                "scores.csv",
                "mated_probabilities.csv",
                "summary.json",
                "mated_probability_comparison.png",
                "mated_probability_comparison.pdf",
                "mated_lid_comparison.png",
                "mated_lid_comparison.pdf",
            }
            for method_name in ("sum_llr", "average_llr"):
                method_dir = output_dir / method_name
                self.assertEqual(
                    {path.name for path in method_dir.iterdir()},
                    method_files,
                )
                method_scores = pd.read_csv(method_dir / "scores.csv")
                self.assertEqual(
                    method_scores.columns.tolist(),
                    [
                        "enroll_spk",
                        "trial_spk",
                        "n_trials",
                        "llr",
                        "ln_p",
                        "p",
                    ],
                )
                np.testing.assert_allclose(
                    method_scores.groupby("trial_spk")["p"].sum(),
                    1.0,
                )
                method_mated = pd.read_csv(
                    method_dir / "mated_probabilities.csv"
                )
                self.assertEqual(set(method_mated["stage"]), {"before", "after"})

            results_dir = Path(temporary_dir) / "results" / "long"
            combined = create_longitudinal_summary(
                [experiment_dir],
                results_dir,
                dpi=40,
            )
            metrics = json.loads((results_dir / "toy" / "results.json").read_text())
            self.assertEqual(
                {
                    "ALID",
                    "PDR",
                    "NDR",
                    "LID+",
                    "LID-",
                    "LID_max",
                }
                - set(metrics),
                set(),
            )
            self.assertEqual(
                set(metrics["before_aggregation"]),
                {"n_trials", "ALID", "PDR", "NDR", "LID+", "LID-", "LID_max"},
            )
            self.assertEqual(
                set(metrics["after_aggregation"]),
                {"n_trials", "ALID", "PDR", "NDR", "LID+", "LID-", "LID_max"},
            )
            self.assertEqual(
                set(metrics["change_after_minus_before"]),
                {"ALID", "PDR", "NDR", "LID+", "LID-", "LID_max"},
            )
            self.assertEqual(metrics["before_aggregation"]["n_trials"], 3)
            self.assertEqual(metrics["after_aggregation"]["n_trials"], 2)
            self.assertAlmostEqual(
                metrics["change_after_minus_before"]["ALID"],
                metrics["after_aggregation"]["ALID"]
                - metrics["before_aggregation"]["ALID"],
            )
            self.assertTrue(
                (results_dir / "toy" / "lid_metrics_before_after.png").is_file()
            )
            self.assertTrue(
                (results_dir / "toy" / "lid_metrics_before_after.pdf").is_file()
            )
            self.assertEqual(combined["experiments"], 1)
            self.assertTrue((results_dir / "summary_table.csv").is_file())
            self.assertTrue((results_dir / "lid_combined_ccdf.png").is_file())
            self.assertTrue(
                (results_dir / "lid_original_vs_averaged_ccdf.png").is_file()
            )
            self.assertTrue(
                (
                    results_dir / "lid_metrics_experiment_comparison.png"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
