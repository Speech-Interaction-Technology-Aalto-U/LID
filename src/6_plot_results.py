import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_paths import calibration_parameters_path, lid_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"
METRICS_FILE = "results.json"
ALTERNATIVE_METRICS_DIR_NAME = "alternative_metrics"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summary"
SEPARATOR = "-" * 72


def relative_path(path):
    return path.relative_to(PROJECT_ROOT)


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.figsize": (3.5, 2.5),
            "figure.dpi": 300,
        }
    )


def plot_metric(
    df,
    column,
    title,
    xlabel,
    out_dir,
    filename_base,
    baseline_val=None,
    baseline_label=None,
    loc="upper right",
):
    scores = df[column].values
    valid_scores = scores[np.isfinite(scores)]
    if len(valid_scores) == 0:
        raise ValueError(f"No finite values found in column: {column}")

    fig, ax = plt.subplots()

    weights = np.ones_like(valid_scores) / len(valid_scores)

    ax.hist(
        valid_scores,
        bins=50,
        weights=weights,
        alpha=0.4,
        color="black",
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Relative Frequency")
    ax.set_ylim(bottom=0)

    if baseline_val is not None:
        ax.axvline(
            x=baseline_val,
            color="red",
            linestyle="dotted",
            linewidth=1.5,
            label=baseline_label,
        )
        ax.legend(
            loc=loc,
            framealpha=0.9,
            borderaxespad=0.2,
            borderpad=0.4,
            labelspacing=0.4,
        )

    plt.tight_layout()

    png_path = out_dir / f"{filename_base}.png"
    pdf_path = out_dir / f"{filename_base}.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"saved {relative_path(png_path)}")
    print(f"saved {relative_path(pdf_path)}")


def get_n_enrolments(experiment_dir, df_lid):
    metrics_path = experiment_dir / METRICS_FILE
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text())
        if "n_enrolments" in metrics:
            return int(metrics["n_enrolments"])

    n_vals = (2 ** df_lid["LID"]) / df_lid["p"]
    return int(np.round(np.median(n_vals)))


def load_lid_dataframe(experiment_dir):
    experiment_lid_path = lid_path(experiment_dir)
    if not experiment_lid_path.is_file():
        raise FileNotFoundError(f"Missing required LID file: {experiment_lid_path}")

    df_lid = pd.read_csv(experiment_lid_path)
    required_columns = {"p", "LID"}
    missing_columns = required_columns - set(df_lid.columns)
    if missing_columns:
        raise ValueError(
            f"{experiment_lid_path} is missing required columns: {sorted(missing_columns)}"
        )

    return df_lid


def plot_probability_distribution(experiment_dir, df_lid, out_dir):
    n_enrolments = get_n_enrolments(experiment_dir, df_lid)

    plot_metric(
        df_lid,
        column="p",
        title=f"Target Recognition Probability Distribution ({experiment_dir.name})",
        xlabel="Probability",
        out_dir=out_dir,
        filename_base="probability_distribution",
        baseline_val=1.0 / n_enrolments,
        baseline_label=f"Random Guess (p = 1/{n_enrolments})",
    )


def plot_lid_distribution(experiment_dir, df_lid, out_dir):
    plot_metric(
        df_lid,
        column="LID",
        title=f"Local Information Disclosure Distribution ({experiment_dir.name})",
        xlabel="Local Information Disclosure (bits)",
        out_dir=out_dir,
        filename_base="lid_distribution",
        baseline_val=0.0,
        baseline_label="No information disclosure",
        loc="upper left",
    )


def experiment_plot_style(experiment_name):
    experiment_colors = {
        "B3": "tab:blue",
        "B4": "tab:orange",
        "B5": "tab:green",
        "T10-2": "tab:red",
        "T12-5": "tab:purple",
        "T25-1": "tab:brown",
        "T8-5": "tab:pink",
        "plain": "tab:gray",
        "random": "tab:olive",
    }
    experiment_styles = {
        "B3": "-",
        "B4": "--",
        "B5": "-.",
        "T10-2": ":",
        "T12-5": "-",
        "T25-1": "--",
        "T8-5": "-.",
        "plain": ":",
        "random": "-",
    }

    prefix = experiment_name.split("_", 1)[0]
    return (
        experiment_colors.get(prefix, "black"),
        experiment_styles.get(prefix, "-"),
    )


def plot_combined_lid_ccdf(all_lid_data, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.27, 4))
    ax.grid(True, which="both", ls="-", color="gray", alpha=0.2, zorder=0)

    for experiment_name, scores in sorted(all_lid_data.items()):
        valid_scores = scores[np.isfinite(scores)]
        if len(valid_scores) == 0:
            continue

        sorted_scores = np.sort(valid_scores)
        y_vals = np.arange(len(sorted_scores), 0, -1) / len(sorted_scores) * 100.0
        avg = float(np.mean(valid_scores))
        mx = float(np.max(valid_scores))
        pos_ratio = float(np.mean(valid_scores > 0))
        color, linestyle = experiment_plot_style(experiment_name)

        ax.plot(
            sorted_scores,
            y_vals,
            linewidth=1.5,
            label=f"{experiment_name} (Avg: {avg:.2f}, Max: {mx:.2f}, Pos: {pos_ratio:.1%})",
            color=color,
            linestyle=linestyle,
        )

    # ax.axvline(
    #     x=0.0,
    #     color="red",
    #     linestyle="dotted",
    #     linewidth=1.5,
    #     label="No information disclosure",
    # )
    ax.set_title("Combined Local Information Disclosure CCDF")
    ax.set_xlabel("Local Information Disclosure (bits)")
    ax.set_ylabel("CCDF (%)")
    ax.set_ylim([0.0, 100.0])
    ax.legend(
        loc="lower left",
        framealpha=0.9,
        borderaxespad=0.2,
        borderpad=0.4,
        labelspacing=0.4,
    )

    plt.tight_layout()

    png_path = out_dir / "lid_combined_ccdf.png"
    pdf_path = out_dir / "lid_combined_ccdf.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"saved {relative_path(png_path)}")
    print(f"saved {relative_path(pdf_path)}")


def process_experiment(experiment_dir):
    print(SEPARATOR)
    print(f"Experiment: {experiment_dir.name}")
    print(SEPARATOR)

    df_lid = load_lid_dataframe(experiment_dir)
    out_dir = RESULTS_DIR / experiment_dir.name / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_probability_distribution(experiment_dir, df_lid, out_dir)
    plot_lid_distribution(experiment_dir, df_lid, out_dir)
    print()


def iter_experiment_dirs():
    if not EXPERIMENTS_DIR.is_dir():
        raise FileNotFoundError(f"Missing experiments directory: {EXPERIMENTS_DIR}")

    experiment_dirs = sorted(path for path in EXPERIMENTS_DIR.iterdir() if path.is_dir())
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiment directories found in: {EXPERIMENTS_DIR}")
    return experiment_dirs


def build_summary_table(experiment_dirs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for experiment_dir in experiment_dirs:
        metrics_path = RESULTS_DIR / experiment_dir.name / METRICS_FILE
        alt_metrics_path = (
            RESULTS_DIR / experiment_dir.name / ALTERNATIVE_METRICS_DIR_NAME / METRICS_FILE
        )

        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing required metrics file: {metrics_path}")
        if not alt_metrics_path.is_file():
            raise FileNotFoundError(
                f"Missing required alternative metrics file: {alt_metrics_path}"
            )

        metrics = json.loads(metrics_path.read_text())
        alt_metrics = json.loads(alt_metrics_path.read_text())
        rows.append(
            {
                "Experiment": experiment_dir.name,
                "EER": float(alt_metrics["EER"]),
                "Cllr": float(alt_metrics["cllr"]),
                "ALID": float(metrics["ALID"]),
                "PDR": float(metrics["PDR"]),
                "NDR": float(metrics["NDR"]),
                "LID+": float(metrics["LID+"])
                if metrics["LID+"] is not None
                else np.nan,
                "LID-": float(metrics["LID-"])
                if metrics["LID-"] is not None
                else np.nan,
                "LID_max": float(metrics["LID_max"]),
            }
        )

    df_summary = pd.DataFrame(rows).sort_values("EER").reset_index(drop=True)
    summary_path = out_dir / "summary_table.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"saved {relative_path(summary_path)}")
    return df_summary


def build_calibration_coeff_table(experiment_dirs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for experiment_dir in experiment_dirs:
        experiment_calibration_path = calibration_parameters_path(experiment_dir)
        if not experiment_calibration_path.is_file():
            raise FileNotFoundError(
                f"Missing required calibration file: {experiment_calibration_path}"
            )

        calibration = json.loads(experiment_calibration_path.read_text())
        if "w" not in calibration or "b" not in calibration:
            raise ValueError(
                f"{experiment_calibration_path} is missing required keys: ['w', 'b']"
            )

        rows.append(
            {
                "experiment": experiment_dir.name,
                "w": float(calibration["w"]),
                "b": float(calibration["b"]),
            }
        )

    calibration_path = out_dir / "calibration_coeff.csv"
    pd.DataFrame(rows).to_csv(calibration_path, index=False)
    print(f"saved {relative_path(calibration_path)}")


def plot_summary_visuals(df_summary, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    df_plot = df_summary[df_summary["Experiment"] != "plain"].copy()
    if df_plot.empty:
        raise ValueError("No experiments available for summary plotting after filtering.")

    x_vals = df_plot["EER"]

    fig, ax = plt.subplots(figsize=(5.4, 3.1))

    experiment_colors = {
        "B3": "tab:blue",
        "B4": "tab:orange",
        "B5": "tab:green",
        "T10-2": "tab:red",
        "T12-5": "tab:purple",
        "T25-1": "tab:brown",
        "T8-5": "tab:pink",
        "plain": "tab:gray",
        "random": "tab:olive",
    }

    import matplotlib.lines as mlines

    for _, row in df_plot.iterrows():
        x_val = row["EER"]
        y_d_minus = row["LID-"]
        y_mi = row["ALID"]
        y_d_plus = row["LID+"]
        y_r_max = row["LID_max"]
        experiment = row["Experiment"]

        color = experiment_colors.get(experiment.split("_", 1)[0], "black")

        ax.scatter(x_val, y_d_minus, color=color, s=25, marker="v", zorder=3)
        ax.scatter(x_val, y_mi, color=color, s=25, marker="o", zorder=3)
        ax.scatter(x_val, y_d_plus, color=color, s=25, marker="^", zorder=3)
        ax.scatter(x_val, y_r_max, color=color, s=25, marker="x", zorder=3)

        y_values = [y for y in [y_d_minus, y_mi, y_d_plus, y_r_max] if np.isfinite(y)]
        if y_values:
            ax.plot(
                [x_val, x_val],
                [min(y_values), max(y_values)],
                color=color,
                linestyle="--",
                linewidth=1,
                zorder=2,
                alpha=0.6,
            )

    type_handles = [
        mlines.Line2D(
            [], [], color="gray", marker="x", linestyle="None", markersize=5, label="LID_max"
        ),
        mlines.Line2D(
            [], [], color="gray", marker="^", linestyle="None", markersize=5, label="LID+"
        ),
        mlines.Line2D(
            [], [], color="gray", marker="o", linestyle="None", markersize=5, label="ALID"
        ),
        mlines.Line2D(
            [], [], color="gray", marker="v", linestyle="None", markersize=5, label="LID-"
        ),
    ]

    metric_legend = ax.legend(
        handles=type_handles,
        fontsize=7,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        title="Metric",
        title_fontsize=7,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
    )
    ax.add_artist(metric_legend)

    experiment_handles = [
        mlines.Line2D(
            [],
            [],
            color=experiment_colors.get(experiment.split("_", 1)[0], "black"),
            marker="o",
            linestyle="None",
            markersize=5,
            label=experiment,
        )
        for experiment in df_plot["Experiment"]
    ]

    ax.legend(
        handles=experiment_handles,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="Experiment",
        title_fontsize=7,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        borderaxespad=0.0,
    )

    ax.axhline(0, color="gray", linestyle=":", linewidth=1, zorder=1, alpha=0.5)
    ax.set_xlabel("Equal Error Rate (EER)", fontsize=9)
    ax.set_ylabel("Information Disclosure (bits)", fontsize=9)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.tick_params(axis="both", which="major", labelsize=8)

    ax.set_xlim(max(0, x_vals.min() - 0.05), min(0.55, x_vals.max() + 0.05))

    all_y_vals = pd.concat(
        [
            df_plot["LID-"],
            df_plot["ALID"],
            df_plot["LID+"],
            df_plot["LID_max"],
        ]
    )
    all_y_vals = all_y_vals[np.isfinite(all_y_vals)]
    y_min = float(all_y_vals.min())
    y_max = float(all_y_vals.max())
    y_margin = (y_max - y_min) * 0.15
    if y_margin == 0:
        y_margin = 0.5
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    plt.tight_layout()

    scatter_png_path = out_dir / "eer_vs_infodisc_scatter.png"
    scatter_pdf_path = out_dir / "eer_vs_infodisc_scatter.pdf"
    plt.savefig(scatter_png_path, dpi=300, bbox_inches="tight")
    plt.savefig(scatter_pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"saved {relative_path(scatter_png_path)}")
    print(f"saved {relative_path(scatter_pdf_path)}")


def plot_combined_summary(experiment_dirs):
    all_lid_data = {}
    for experiment_dir in experiment_dirs:
        df_lid = load_lid_dataframe(experiment_dir)
        all_lid_data[experiment_dir.name] = df_lid["LID"].to_numpy()

    plot_combined_lid_ccdf(all_lid_data, SUMMARY_DIR)
    df_summary = build_summary_table(experiment_dirs, SUMMARY_DIR)
    build_calibration_coeff_table(experiment_dirs, SUMMARY_DIR)
    plot_summary_visuals(df_summary, SUMMARY_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot local information disclosure results.")
    parser.add_argument(
        "experiment",
        nargs="?",
        help="Experiment name under data/experiments. If omitted, all experiments are processed.",
    )

    args = parser.parse_args()

    configure_matplotlib()

    print("Plotting local information disclosure results.")
    print("For each experiment, probability and LID histograms are saved as PNG and PDF")
    print("under results/experiments/{experiment}/plots.")
    print("When all experiments are processed, combined summary plots and a metrics table")
    print("are saved under results/experiments/summary.\n")

    if args.experiment:
        process_experiment(EXPERIMENTS_DIR / args.experiment)
    else:
        experiment_dirs = iter_experiment_dirs()
        for experiment_dir in experiment_dirs:
            process_experiment(experiment_dir)
        plot_combined_summary(experiment_dirs)
