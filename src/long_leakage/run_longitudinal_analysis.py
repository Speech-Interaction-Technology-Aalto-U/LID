"""Run longitudinal leakage analysis for calibrated experiments."""

import argparse
import importlib
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from long_leakage.analysis import (
    analyse_experiment,
    create_longitudinal_summary,
    speaker_metric_comparison,
)
from long_leakage.calibration import analyse_calibrated_aggregations
from tools.utils import (
    PROJECT_ROOT,
    iter_experiment_dirs,
    relative_path,
    SEPARATOR,
    SEPARATOR2,
)


def regenerate_speaker_plots(experiment_dir, mated_path, dpi, out_dir=None):
    mated = pd.read_csv(mated_path, dtype={"trial_spk": "string"})
    trial_values = mated[mated["stage"] == "before"].copy()
    comparison = speaker_metric_comparison(mated)

    plot_module = importlib.import_module(
        "pipeline.6_create_plots_and_summaries"
    )
    return plot_module.regenerate_longitudinal_speaker_plots(
        experiment_dir,
        trial_values,
        comparison,
        dpi=dpi,
        out_dir=out_dir,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate calibrated LLR vectors by trial speaker, fit "
            "longitudinal calibration on development speakers, and create "
            "static and interactive leakage reports."
        )
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        metavar="EXPERIMENT",
        help="Only process these experiment directory names.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster plot resolution (default: 300).",
    )
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    print()
    print(SEPARATOR)
    print("Longitudinal leakage analysis")
    print(SEPARATOR)

    experiment_dirs = iter_experiment_dirs(args.experiments)
    summaries = []
    for experiment_dir in experiment_dirs:
        summary = analyse_experiment(experiment_dir, dpi=args.dpi)
        calibrated_summary = analyse_calibrated_aggregations(
            experiment_dir,
            dpi=args.dpi,
        )
        speaker_plot_paths = regenerate_speaker_plots(
            experiment_dir,
            summary["mated_path"],
            dpi=args.dpi,
        )
        summary["speaker_plot_paths"] = speaker_plot_paths
        summary["calibrated_aggregation"] = calibrated_summary
        summaries.append(summary)
        count_parameters = calibrated_summary["fitted_count_parameters"]
        similarity_parameters = calibrated_summary[
            "fitted_similarity_parameters"
        ]
        print(
            f"{experiment_dir.name}: {summary['n_trials']} trials from "
            f"{summary['n_trial_speakers']} speakers; wrote "
            f"{relative_path(experiment_dir / 'long')} and regenerated "
            f"{len(speaker_plot_paths)} by-speaker plot files; development "
            f"fits: global T(NLL)="
            f"{calibrated_summary['fitted_temperature']:.6g}, "
            f"global T(Brier)="
            f"{calibrated_summary['fitted_brier_temperature']:.6g}, "
            f"count (T={count_parameters['temperature']:.6g}, "
            f"rho={count_parameters['rho']:.6g}), embedding "
            f"(T={similarity_parameters['temperature']:.6g}, "
            f"rho={similarity_parameters['rho']:.6g})"
        )

    results_dir = PROJECT_ROOT / "results" / "long"
    create_longitudinal_summary(experiment_dirs, results_dir, dpi=args.dpi)
    print(f"Combined metrics and plots: {relative_path(results_dir)}")
    print()
    print(
        f"Processed {len(summaries)} experiment"
        f"{'' if len(summaries) == 1 else 's'}."
    )
    print(SEPARATOR2)
    print()


if __name__ == "__main__":
    main()
