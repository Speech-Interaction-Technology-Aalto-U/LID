from itertools import combinations
import json
from math import comb
from pathlib import Path
import random
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
from long_leakage.run_combination_convergence import (
    METHODS,
    _lid_from_logits,
    _sample_unique_ranks,
    _unrank_combination,
    analyse_experiment_convergence,
    combination_lid_populations,
    load_fitted_temperatures,
    summarize_populations,
)


def convergence_scores():
    vectors = [
        ("a-1", "A", [2.0, 0.0]),
        ("a-2", "A", [1.0, 0.5]),
        ("a-3", "A", [-0.5, 1.0]),
        ("b-1", "B", [0.2, 1.7]),
        ("b-2", "B", [1.1, 0.8]),
    ]
    rows = []
    for trial_id, trial_spk, llrs in vectors:
        for enroll_spk, llr in zip(("A", "B"), llrs):
            rows.append(
                {
                    "enroll_spk": enroll_spk,
                    "trial_spk": trial_spk,
                    "trial_id": trial_id,
                    "llr": llr,
                }
            )
    return pd.DataFrame(rows)


def write_fitted_temperatures(experiment_dir, nll=2.0, brier=3.0):
    long_dir = experiment_dir / "long"
    long_dir.mkdir(exist_ok=True)
    calibration_path = long_dir / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "methods": {
                    "temperature_scaled": {
                        "fitted_parameters": {"temperature": nll}
                    },
                    "temperature_scaled_brier": {
                        "fitted_parameters": {"temperature": brier}
                    },
                }
            }
        )
    )
    return calibration_path


class LongitudinalConvergenceTests(unittest.TestCase):

    def test_combination_unranking_covers_lexicographic_population(self):
        expected = list(combinations(range(5), 3))
        actual = [
            _unrank_combination(5, 3, rank)
            for rank in range(comb(5, 3))
        ]
        self.assertEqual(actual, expected)

    def test_unique_rank_sampling_supports_large_integer_populations(self):
        population_size = 10**25
        ranks = _sample_unique_ranks(
            population_size,
            100,
            random.Random(7),
        )
        self.assertEqual(len(ranks), 100)
        self.assertEqual(len(set(ranks)), 100)
        self.assertGreaterEqual(min(ranks), 0)
        self.assertLess(max(ranks), population_size)

    def test_exact_populations_use_all_within_speaker_subsets(self):
        evidence = build_longitudinal_evidence(convergence_scores())
        populations = combination_lid_populations(
            evidence,
            nll_temperature=2.0,
            brier_temperature=3.0,
            max_combinations=100,
        )

        self.assertEqual(
            [population.total_combinations for population in populations],
            [5, comb(3, 2) + comb(2, 2), 1],
        )
        self.assertTrue(all(population.is_exact for population in populations))
        self.assertEqual(
            [population.eligible_speakers for population in populations],
            [2, 2, 1],
        )

        first_trial_logits = np.asarray([[2.0, 0.0]])
        expected_first_lid = _lid_from_logits(first_trial_logits, 0, 2)[0]
        self.assertAlmostEqual(
            populations[0].values["sum_llr"][0],
            expected_first_lid,
        )
        np.testing.assert_allclose(
            populations[0].values["sum_llr"],
            populations[0].values["average_llr"],
        )

        a_all_sum = np.asarray([[2.5, 1.5]])
        expected_summed = _lid_from_logits(a_all_sum, 0, 2)[0]
        expected_averaged = _lid_from_logits(a_all_sum / 3.0, 0, 2)[0]
        expected_nll = _lid_from_logits(a_all_sum / 2.0, 0, 2)[0]
        self.assertAlmostEqual(
            populations[2].values["sum_llr"][0],
            expected_summed,
        )
        self.assertAlmostEqual(
            populations[2].values["average_llr"][0],
            expected_averaged,
        )
        self.assertAlmostEqual(
            populations[2].values["temperature_scaled_nll"][0],
            expected_nll,
        )
        self.assertAlmostEqual(
            populations[2].values["temperature_scaled_brier"][0],
            expected_averaged,
        )
        self.assertEqual(set(populations[0].values), set(METHODS))

    def test_sampling_is_reproducible_and_reported_as_sampled(self):
        evidence = build_longitudinal_evidence(convergence_scores())
        first = combination_lid_populations(
            evidence,
            nll_temperature=2.0,
            brier_temperature=3.0,
            max_combinations=2,
            seed=19,
            max_k=2,
        )
        second = combination_lid_populations(
            evidence,
            nll_temperature=2.0,
            brier_temperature=3.0,
            max_combinations=2,
            seed=19,
            max_k=2,
        )

        for left, right in zip(first, second):
            self.assertFalse(left.is_exact)
            self.assertEqual(left.evaluated_combinations, 2)
            for method in METHODS:
                np.testing.assert_array_equal(
                    left.values[method],
                    right.values[method],
                )

        summary = summarize_populations(
            "toy",
            {"nll": 2.0, "brier": 3.0},
            first,
        )
        self.assertEqual(set(summary["evaluation"]), {"sampled"})
        self.assertEqual(set(summary["method"]), set(METHODS))
        self.assertTrue(
            (summary["mean_95ci_high_bits"] >= summary["mean_95ci_low_bits"]).all()
        )

        one_draw = combination_lid_populations(
            evidence,
            nll_temperature=2.0,
            brier_temperature=3.0,
            max_combinations=1,
            max_k=1,
        )
        one_draw_summary = summarize_populations(
            "toy",
            {"nll": 2.0, "brier": 3.0},
            one_draw,
        )
        self.assertTrue(
            np.isfinite(one_draw_summary["variance_LID_bits2"]).all()
        )

    def test_writes_clear_statistics_metadata_and_plots(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            experiment_dir = root / "toy"
            scores_dir = experiment_dir / "scores"
            scores_dir.mkdir(parents=True)
            convergence_scores().to_csv(
                scores_dir / "test_scores.csv",
                index=False,
            )
            write_fitted_temperatures(experiment_dir)

            result = analyse_experiment_convergence(
                experiment_dir,
                root / "results",
                max_combinations=100,
                dpi=40,
            )

            output_dir = root / "results" / "toy"
            self.assertTrue(result["summary_path"].is_file())
            self.assertTrue(result["metadata_path"].is_file())
            self.assertEqual(
                set(result["summary"]["method"]),
                set(METHODS),
            )
            for plot_path in result["plot_paths"]:
                self.assertTrue(plot_path.is_file())
                self.assertGreater(plot_path.stat().st_size, 0)
            metadata = json.loads(result["metadata_path"].read_text())
            self.assertEqual(
                metadata["temperatures"],
                {"nll": 2.0, "brier": 3.0},
            )
            self.assertIn(
                "T = k",
                metadata["aggregation_rules"]["average_llr"],
            )

    def test_loads_and_uses_both_development_fitted_temperatures(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            experiment_dir = root / "toy"
            scores_dir = experiment_dir / "scores"
            scores_dir.mkdir(parents=True)
            convergence_scores().to_csv(
                scores_dir / "test_scores.csv",
                index=False,
            )
            calibration_path = write_fitted_temperatures(experiment_dir)

            temperatures, path = load_fitted_temperatures(experiment_dir)
            self.assertEqual(temperatures, {"nll": 2.0, "brier": 3.0})
            self.assertEqual(path, calibration_path)

            result = analyse_experiment_convergence(
                experiment_dir,
                root / "results",
                max_combinations=100,
                dpi=40,
            )
            self.assertEqual(result["temperatures"], temperatures)
            self.assertEqual(
                set(result["summary"]["method"]),
                set(METHODS),
            )


if __name__ == "__main__":
    unittest.main()
