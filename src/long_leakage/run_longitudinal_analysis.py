"""Run longitudinal leakage analysis for calibrated experiments."""

import argparse
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from long_leakage.analysis import analyse_experiment, create_longitudinal_summary
from tools.utils import (
    PROJECT_ROOT,
    iter_experiment_dirs,
    relative_path,
    SEPARATOR,
    SEPARATOR2,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sum calibrated test LLR vectors by trial speaker and create "
            "longitudinal leakage data and plots."
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
        summaries.append(summary)
        print(
            f"{experiment_dir.name}: {summary['n_trials']} trials from "
            f"{summary['n_trial_speakers']} speakers; wrote "
            f"{relative_path(experiment_dir / 'long')}"
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
