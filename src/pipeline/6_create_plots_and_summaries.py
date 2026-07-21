"""
Create experiment plots and summary tables for local information disclosure results.

For each experiment, this script plots the test-trial probability and LID
distributions. Across experiments, it writes a combined LID CCDF, a summary
metrics table, calibration coefficients, and, when EER values are available, an
EER-vs-information-disclosure scatter plot.

Inputs:
    <experiment>/outputs/local_information_disclosures.csv
    <experiment>/outputs/calibration_parameters.json
    results/experiments/<experiment>/results.json
    results/experiments/<experiment>/alternative_metrics/results.json, optional

Outputs:
    results/experiments/<experiment>/plots/*.png and *.pdf
    results/summary/summary_table.csv
    results/summary/calibration_coeff.csv
    results/summary/lid_combined_ccdf.png and .pdf
    results/summary/eer_vs_infodisc_scatter.png and .pdf, when EER is available
"""

import json
from pathlib import Path
import shutil
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.utils import (
    PROJECT_ROOT,
    calibration_parameters_path,
    iter_experiment_dirs,
    lid_path,
    SEPARATOR,
    SEPARATOR2,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"
METRICS_FILE = "results.json"
ALTERNATIVE_METRICS_DIR_NAME = "alternative_metrics"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summary"
PAPER_FIGURES_DIR = PROJECT_ROOT / "results" / "paper_figures"


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

    return [png_path, pdf_path]


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

    return plot_metric(
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
    return plot_metric(
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
    ax.set_ylabel("Trials Exceeding Disclosure Level (%)")
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

    return [png_path, pdf_path]


def process_experiment(experiment_dir):
    df_lid = load_lid_dataframe(experiment_dir)
    out_dir = RESULTS_DIR / experiment_dir.name / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = []
    plot_paths.extend(plot_probability_distribution(experiment_dir, df_lid, out_dir))
    plot_paths.extend(plot_lid_distribution(experiment_dir, df_lid, out_dir))

    return {
        "experiment": experiment_dir.name,
        "plots": len(plot_paths),
    }


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

        metrics = json.loads(metrics_path.read_text())
        row = {
            "Experiment": experiment_dir.name,
            "ALID": float(metrics["ALID"]),
            "PDR": float(metrics["PDR"]),
            "NDR": float(metrics["NDR"]),
            "LID+": float(metrics["LID+"]) if metrics["LID+"] is not None else np.nan,
            "LID-": float(metrics["LID-"]) if metrics["LID-"] is not None else np.nan,
            "LID_max": float(metrics["LID_max"]),
        }

        if alt_metrics_path.is_file():
            alt_metrics = json.loads(alt_metrics_path.read_text())
            if "EER" in alt_metrics and alt_metrics["EER"] is not None:
                row["EER"] = float(alt_metrics["EER"])
            if "cllr" in alt_metrics and alt_metrics["cllr"] is not None:
                row["Cllr"] = float(alt_metrics["cllr"])

        rows.append(row)

    df_summary = pd.DataFrame(rows)
    optional_columns = ["EER", "Cllr"]
    empty_optional_columns = [
        column
        for column in optional_columns
        if column in df_summary and not df_summary[column].notna().any()
    ]
    df_summary = df_summary.drop(columns=empty_optional_columns)

    if "EER" in df_summary:
        df_summary = df_summary.sort_values("EER", na_position="last")
    else:
        df_summary = df_summary.sort_values("Experiment")
    df_summary = df_summary.reset_index(drop=True)

    ordered_columns = [
        column
        for column in [
            "Experiment",
            "EER",
            "Cllr",
            "ALID",
            "PDR",
            "NDR",
            "LID+",
            "LID-",
            "LID_max",
        ]
        if column in df_summary
    ]
    df_summary = df_summary[ordered_columns]

    summary_path = out_dir / "summary_table.csv"
    df_summary.to_csv(summary_path, index=False)
    return df_summary, summary_path


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
    return calibration_path


def plot_summary_visuals(df_summary, out_dir):
    if "EER" not in df_summary:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    df_plot = df_summary[
        (df_summary["Experiment"] != "plain") & df_summary["EER"].notna()
    ].copy()
    if df_plot.empty:
        return []

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

    return [scatter_png_path, scatter_pdf_path]


def format_table_value(value):
    if pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_dataframe_table(df):
    rows = [
        [format_table_value(row[column]) for column in df.columns]
        for _, row in df.iterrows()
    ]
    widths = [
        max(len(column), *(len(row[index]) for row in rows))
        for index, column in enumerate(df.columns)
    ]

    header_row = "  ".join(
        column.ljust(widths[index]) for index, column in enumerate(df.columns)
    )
    separator_row = "  ".join("-" * width for width in widths)
    print(header_row)
    print(separator_row)
    for row in rows:
        print(
            "  ".join(
                value.rjust(widths[index]) if index else value.ljust(widths[index])
                for index, value in enumerate(row)
            )
        )


def copy_png_figure(source_path, figure_name, out_dir, required=True):
    source = source_path.with_suffix(".png")
    if not source.is_file():
        if required:
            raise FileNotFoundError(f"Missing required figure source: {source}")
        return None

    destination = out_dir / f"{figure_name}.png"
    shutil.copy2(source, destination)
    return destination


def save_rounded_table(source_csv_path, table_name, out_dir):
    if not source_csv_path.is_file():
        raise FileNotFoundError(f"Missing required table source: {source_csv_path}")

    destination = out_dir / f"{table_name}.csv"
    df = pd.read_csv(source_csv_path)
    numeric_columns = df.select_dtypes(include="number").columns
    df[numeric_columns] = df[numeric_columns].mask(
        df[numeric_columns].abs() < 0.005,
        0.0,
    )
    df.round(2).to_csv(destination, index=False, float_format="%.2f")
    return destination


def export_paper_figures():
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    t10_plots_dir = RESULTS_DIR / "T10-2" / "plots"
    exported_paths = []
    skipped = []

    exported_paths.append(
        copy_png_figure(
            t10_plots_dir / "probability_distribution.pdf",
            "Figure2",
            PAPER_FIGURES_DIR,
        )
    )
    exported_paths.append(
        copy_png_figure(
            t10_plots_dir / "lid_distribution.png",
            "Figure3",
            PAPER_FIGURES_DIR,
        )
    )
    exported_paths.append(
        copy_png_figure(
            SUMMARY_DIR / "lid_combined_ccdf.png",
            "Figure4",
            PAPER_FIGURES_DIR,
        )
    )

    figure5_path = copy_png_figure(
        SUMMARY_DIR / "eer_vs_infodisc_scatter.png",
        "Figure5",
        PAPER_FIGURES_DIR,
        required=False,
    )
    if figure5_path:
        exported_paths.append(figure5_path)
    else:
        skipped.append("Figure5")

    exported_paths.append(
        save_rounded_table(
            SUMMARY_DIR / "calibration_coeff.csv",
            "Table3",
            PAPER_FIGURES_DIR,
        )
    )
    exported_paths.append(
        save_rounded_table(
            SUMMARY_DIR / "summary_table.csv",
            "Table4",
            PAPER_FIGURES_DIR,
        )
    )

    return {
        "files": len(exported_paths),
        "skipped": skipped,
    }


def plot_combined_summary(experiment_dirs):
    all_lid_data = {}
    for experiment_dir in experiment_dirs:
        df_lid = load_lid_dataframe(experiment_dir)
        all_lid_data[experiment_dir.name] = df_lid["LID"].to_numpy()

    summary_paths = []
    summary_paths.extend(plot_combined_lid_ccdf(all_lid_data, SUMMARY_DIR))
    df_summary, summary_table_path = build_summary_table(experiment_dirs, SUMMARY_DIR)
    summary_paths.append(summary_table_path)
    summary_paths.append(build_calibration_coeff_table(experiment_dirs, SUMMARY_DIR))
    eer_plot_paths = plot_summary_visuals(df_summary, SUMMARY_DIR)
    summary_paths.extend(eer_plot_paths)
    paper_export_summary = export_paper_figures()

    return {
        "summary_table": df_summary,
        "summary_files": len(summary_paths),
        "eer_plot_created": bool(eer_plot_paths),
        "paper_files": paper_export_summary["files"],
        "paper_skipped": paper_export_summary["skipped"],
    }


if __name__ == "__main__":
    configure_matplotlib()

    print()
    print(SEPARATOR)
    print("STEP 6. Plotting local information disclosure results.")
    print(SEPARATOR)
    print(
        "Creating probability/LID histograms, combined summaries, and paper artifacts."
    )
    print()

    experiment_dirs = iter_experiment_dirs()
    experiment_summaries = []
    for experiment_dir in experiment_dirs:
        experiment_summaries.append(process_experiment(experiment_dir))
    summary = plot_combined_summary(experiment_dirs)

    total_experiment_plots = sum(item["plots"] for item in experiment_summaries)
    print(f"Generated {total_experiment_plots} per-experiment plot files.")
    print(f"Generated {summary['summary_files']} summary files.")
    print(f"Exported {summary['paper_files']} paper figure/table files.")
    if not summary["eer_plot_created"]:
        print("Skipped EER vs information disclosure plot because EER is unavailable.")
    for item in summary["paper_skipped"]:
        print(f"Skipped {item} paper export because its source figure is unavailable.")
    print()
    print_dataframe_table(summary["summary_table"])
    print()
    print("Plots are saved in results/experiments/<experiment>/plots/.")
    print("Summary outputs can be found in results/summary/.")
    print("Paper artifacts can be found in results/paper_figures/.")
    print(SEPARATOR2)
    print()
