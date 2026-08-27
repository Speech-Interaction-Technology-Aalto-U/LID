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
    results/summary/Table3.csv and Table4.csv
    results/summary/lid_combined_ccdf.png and .pdf
    results/summary/eer_vs_infodisc_scatter.png and .pdf, when EER is available
    results/readme_figures/Figure2.png, Figure3.png, and Figure4.png
    results/paper_figures/Figure2.pdf through Figure5.pdf, without plot titles
    results/paper_figures/Figure4_no_legend.pdf and Figure5_no_legend.pdf
    results/paper_figures/Table3.tex and Table4.tex
"""

import json
import shutil

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
README_FIGURES_DIR = PROJECT_ROOT / "results" / "readme_figures"
# MeanD is not computed by this pipeline. It is sourced from previous research (Evaluating voice anonymisation using similarity rank disclosure by Chandra et al.)
MEAND_BY_EXPERIMENT = {
    "B3": 0.06,
    "B4": 0.02,
    "B5": 0.02,
    "T8-5": 0.06,
    "T10-2": 0.52,
    "T12-5": 0.02,
    "T25-1": 0.05,
    "plain": 2.19,
    "random": 0.00,
}
PAPER_SYSTEM_ORDER = [
    "B3",
    "B4",
    "B5",
    "T8-5",
    "T10-2",
    "T12-5",
    "T25-1",
]
BASELINE_EXPERIMENT_ORDER = ["plain", "random"]

TABLE4_COLUMN_CONFIG = {
    "EER": {
        "header": r"\textbf{EER} ($\uparrow$)",
        "unit": "",
        "decimals": 2,
        "direction": "max",
    },
    "Cllr": {
        "header": r"$\mathbf{C}_{\text{llr}}$ ($\uparrow$)",
        "unit": "",
        "decimals": 2,
        "direction": "max",
    },
    "MeanD": {
        "header": r"\textbf{MeanD} ($\downarrow$)",
        "unit": r"\textbf{(bits)}",
        "decimals": 2,
        "direction": "min",
    },
    "ALID": {
        "header": r"\textbf{ALID} ($\downarrow$)",
        "unit": r"\textbf{(bits)}",
        "decimals": 2,
        "direction": "min",
    },
    "PDR": {
        "header": r"\textbf{PDR} ($\downarrow$)",
        "unit": r"\textbf{(\%)}",
        "decimals": 0,
        "direction": "min",
    },
    "NDR": {
        "header": r"\textbf{NDR} ($\uparrow$)",
        "unit": r"\textbf{(\%)}",
        "decimals": 0,
        "direction": "max",
    },
    "LID+": {
        "header": r"$\mathbf{LID}^{+}$ ($\downarrow$)",
        "unit": r"\textbf{(bits)}",
        "decimals": 2,
        "direction": "min",
    },
    "LID-": {
        "header": r"$\mathbf{LID}^{-}$ ($\downarrow$)",
        "unit": r"\textbf{(bits)}",
        "decimals": 2,
        "direction": "min",
    },
    "LID_max": {
        "header": r"$\mathbf{LID}_{\text{max}}$ ($\downarrow$)",
        "unit": r"\textbf{(bits)}",
        "decimals": 2,
        "direction": "min",
    },
}
LEGEND_STYLE = {
    "frameon": True,
    "framealpha": 0.9,
    "borderaxespad": 0.2,
    "borderpad": 0.4,
    "labelspacing": 0.4,
}


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def create_metric_figure(
    df,
    column,
    title,
    xlabel,
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

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Proportion of trials")
    ax.set_ylim(bottom=0)

    if baseline_val is not None:
        ax.axvline(
            x=baseline_val,
            color="red",
            linestyle="dotted",
            linewidth=1.5,
            label=baseline_label,
        )
        ax.legend(loc=loc, **LEGEND_STYLE)

    plt.tight_layout()
    return fig


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
    fig = create_metric_figure(
        df,
        column,
        title,
        xlabel,
        baseline_val=baseline_val,
        baseline_label=baseline_label,
        loc=loc,
    )

    png_path = out_dir / f"{filename_base}.png"
    pdf_path = out_dir / f"{filename_base}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, format="pdf")
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
        title=f"Attacker Confidence Distribution ({experiment_dir.name})",
        xlabel=r"Probability assigned to true identity, $p_{i,m_i}$",
        out_dir=out_dir,
        filename_base="probability_distribution",
        baseline_val=1.0 / n_enrolments,
        baseline_label=f"Random Guess (p = 1/{n_enrolments})",
    )


def plot_lid_distribution(experiment_dir, df_lid, out_dir):
    return plot_metric(
        df_lid,
        column="LID",
        title=f"LID Distribution ({experiment_dir.name})",
        xlabel=r"Local information disclosure, $\mathrm{LID}_i$ (bits)",
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


def create_combined_lid_ccdf_figure(
    all_lid_data,
    title="Local Information Disclosure Across Systems",
    show_legend=True,
):
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

        ax.step(
            sorted_scores,
            y_vals,
            where="post",  # A step function accurately represents the empirical CCDF.
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
    if title:
        ax.set_title(title)
    ax.set_xlabel(r"Local information disclosure, $\mathrm{LID}_i$ (bits)")
    ax.set_ylabel("Trials exceeding disclosure level (%)")
    ax.set_ylim([0.0, 100.0])
    experiment_legend = None
    if show_legend:
        experiment_legend = ax.legend(
            loc="lower left",
            framealpha=0.9,
            borderaxespad=0.2,
            borderpad=0.4,
            labelspacing=0.4,
        )

    plt.tight_layout()
    return fig, experiment_legend


def plot_combined_lid_ccdf(all_lid_data, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, experiment_legend = create_combined_lid_ccdf_figure(all_lid_data)

    png_path = out_dir / "lid_combined_ccdf.png"
    pdf_path = out_dir / "lid_combined_ccdf.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

    experiment_legend.remove()
    no_legend_png_path = out_dir / "lid_combined_ccdf_no_legend.png"
    no_legend_pdf_path = out_dir / "lid_combined_ccdf_no_legend.pdf"
    plt.tight_layout()
    plt.savefig(no_legend_png_path, dpi=300, bbox_inches="tight")
    plt.savefig(no_legend_pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return [png_path, pdf_path, no_legend_png_path, no_legend_pdf_path]


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

        if experiment_dir.name in MEAND_BY_EXPERIMENT:
            row["MeanD"] = MEAND_BY_EXPERIMENT[experiment_dir.name]

        if alt_metrics_path.is_file():
            alt_metrics = json.loads(alt_metrics_path.read_text())
            if "EER" in alt_metrics and alt_metrics["EER"] is not None:
                row["EER"] = float(alt_metrics["EER"])
            if "cllr" in alt_metrics and alt_metrics["cllr"] is not None:
                row["Cllr"] = float(alt_metrics["cllr"])

        rows.append(row)

    df_summary = pd.DataFrame(rows)
    optional_columns = ["EER", "Cllr", "MeanD"]
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
            "MeanD",
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

    df_calibration = pd.DataFrame(rows)
    calibration_path = out_dir / "calibration_coeff.csv"
    df_calibration.to_csv(calibration_path, index=False)
    return df_calibration, calibration_path


def plot_summary_visuals(df_summary, out_dir):
    if "EER" not in df_summary:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)

    df_plot = df_summary[
        (df_summary["Experiment"] != "plain") & df_summary["EER"].notna()
    ].copy()
    if df_plot.empty:
        return []

    x_vals = df_plot["EER"]

    fig, ax = plt.subplots()

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
            [], [], color="gray", marker="x", linestyle="None", markersize=5, label=r"$\mathrm{LID}_{\max}$"
        ),
        mlines.Line2D(
            [], [], color="gray", marker="^", linestyle="None", markersize=5, label=r"$\mathrm{LID}^{+}$"
        ),
        mlines.Line2D(
            [], [], color="gray", marker="o", linestyle="None", markersize=5, label=r"$\mathrm{ALID}$"
        ),
        mlines.Line2D(
            [], [], color="gray", marker="v", linestyle="None", markersize=5, label=r"$\mathrm{LID}^{-}$"
        ),
    ]

    metric_legend = ax.legend(
        handles=type_handles,
        loc="upper right",
        title="Metric",
        **LEGEND_STYLE,
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

    experiment_legend = ax.legend(
        handles=experiment_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="Experiment",
        **LEGEND_STYLE,
    )

    ax.axhline(0, color="gray", linestyle=":", linewidth=1, zorder=1, alpha=0.5)
    ax.set_xlabel("Equal error rate (EER)")
    ax.set_ylabel("Information disclosure (bits)")
    ax.set_axisbelow(True)
    ax.grid(True, which="both", color="0.85", linewidth=0.6, alpha=0.7)

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
    fig.savefig(scatter_png_path, dpi=300)
    fig.savefig(scatter_pdf_path, format="pdf")

    experiment_legend.remove()
    no_legend_png_path = out_dir / "eer_vs_infodisc_scatter_no_legend.png"
    no_legend_pdf_path = out_dir / "eer_vs_infodisc_scatter_no_legend.pdf"
    plt.tight_layout()
    fig.savefig(no_legend_png_path, dpi=300)
    fig.savefig(no_legend_pdf_path, format="pdf")
    plt.close(fig)

    return [
        scatter_png_path,
        scatter_pdf_path,
        no_legend_png_path,
        no_legend_pdf_path,
    ]


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


def copy_figure(source_path, figure_name, out_dir, required=True):
    if not source_path.is_file():
        if required:
            raise FileNotFoundError(f"Missing required figure source: {source_path}")
        return None

    destination = out_dir / f"{figure_name}{source_path.suffix}"
    shutil.copy2(source_path, destination)
    return destination


def paper_experiment_sort_key(experiment_name):
    if experiment_name in PAPER_SYSTEM_ORDER:
        return (0, PAPER_SYSTEM_ORDER.index(experiment_name), "")
    if experiment_name in BASELINE_EXPERIMENT_ORDER:
        return (2, BASELINE_EXPERIMENT_ORDER.index(experiment_name), "")
    return (1, 0, experiment_name.lower())


def order_paper_table_rows(df, experiment_column):
    row_order = sorted(
        range(len(df)),
        key=lambda index: paper_experiment_sort_key(df.iloc[index][experiment_column]),
    )
    return df.iloc[row_order].reset_index(drop=True)


def prepare_table3_data(df_calibration):
    df_table = df_calibration.rename(columns={"experiment": "Experiment"}).copy()
    df_table = order_paper_table_rows(df_table, "Experiment")
    return df_table[["Experiment", "w", "b"]]


def prepare_table4_data(df_summary):
    metric_columns = [
        column for column in TABLE4_COLUMN_CONFIG if column in df_summary.columns
    ]
    df_table = df_summary[["Experiment", *metric_columns]].copy()
    df_table = order_paper_table_rows(df_table, "Experiment")
    for column in ["PDR", "NDR"]:
        if column in df_table:
            df_table[column] *= 100.0
    return df_table


def normalized_table_value(value, decimals):
    if pd.isna(value):
        return np.nan
    rounded_value = round(float(value), decimals)
    if rounded_value == 0:
        return 0.0
    return rounded_value


def format_table_number(value, column):
    if pd.isna(value):
        return ""
    decimals = TABLE4_COLUMN_CONFIG[column]["decimals"]
    normalized_value = normalized_table_value(value, decimals)
    if column in {"PDR", "NDR"}:
        return f"{normalized_value:.1f}"
    return f"{normalized_value:.2f}"


def write_table3_csv(df_table, destination):
    df_csv = df_table.copy()
    for column in ["w", "b"]:
        df_csv[column] = df_csv[column].map(
            lambda value: f"{normalized_table_value(value, 2):.2f}"
        )
    df_csv.to_csv(destination, index=False)
    return destination


def write_table4_csv(df_table, destination):
    df_csv = df_table.copy()
    for column in df_csv.columns[1:]:
        df_csv[column] = df_csv[column].map(
            lambda value, metric=column: format_table_number(value, metric)
        )
    df_csv.to_csv(destination, index=False)
    return destination


def latex_escape(value):
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def latex_row(values):
    return " & ".join(values) + r" \\"


def write_table3_latex(df_table, destination):
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        latex_row([r"\textbf{Experiment}", r"$\mathbf{w}$", r"$\mathbf{b}$"]),
        r"\midrule",
    ]
    baseline_started = False
    for _, row in df_table.iterrows():
        is_baseline = row["Experiment"] in BASELINE_EXPERIMENT_ORDER
        if is_baseline and not baseline_started:
            lines.append(r"\midrule")
            baseline_started = True
        lines.append(
            latex_row(
                [
                    latex_escape(row["Experiment"]),
                    f"{normalized_table_value(row['w'], 2):.2f}",
                    f"{normalized_table_value(row['b'], 2):.2f}",
                ]
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    destination.write_text("\n".join(lines) + "\n")
    return destination


def table4_best_values(df_table):
    experiment_rows = df_table[
        ~df_table["Experiment"].isin(BASELINE_EXPERIMENT_ORDER)
    ]
    best_values = {}
    for column in df_table.columns[1:]:
        config = TABLE4_COLUMN_CONFIG[column]
        values = experiment_rows[column].dropna().map(
            lambda value: normalized_table_value(value, config["decimals"])
        )
        if values.empty:
            continue
        if config["direction"] == "max":
            best_values[column] = values.max()
        else:
            best_values[column] = values.min()
    return best_values


def write_table4_latex(df_table, destination):
    metric_columns = list(df_table.columns[1:])
    lines = [
        rf"\begin{{tabular}}{{l{'c' * len(metric_columns)}}}",
        r"\toprule",
        latex_row(
            [r"\textbf{Experiment}"]
            + [TABLE4_COLUMN_CONFIG[column]["header"] for column in metric_columns]
        ),
        latex_row(
            [""] + [TABLE4_COLUMN_CONFIG[column]["unit"] for column in metric_columns]
        ),
        r"\midrule",
    ]
    best_values = table4_best_values(df_table)
    baseline_started = False
    for _, row in df_table.iterrows():
        is_baseline = row["Experiment"] in BASELINE_EXPERIMENT_ORDER
        if is_baseline and not baseline_started:
            lines.append(r"\midrule")
            baseline_started = True

        formatted_values = []
        for column in metric_columns:
            if pd.isna(row[column]):
                formatted_values.append("-")
                continue
            value = normalized_table_value(
                row[column], TABLE4_COLUMN_CONFIG[column]["decimals"]
            )
            formatted_value = format_table_number(row[column], column)
            if not is_baseline and value == best_values.get(column):
                formatted_value = rf"\textbf{{{formatted_value}}}"
            formatted_values.append(formatted_value)

        lines.append(
            latex_row([latex_escape(row["Experiment"]), *formatted_values])
        )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    destination.write_text("\n".join(lines) + "\n")
    return destination


def export_tables(df_calibration, df_summary):
    table3 = prepare_table3_data(df_calibration)
    table4 = prepare_table4_data(df_summary)

    summary_paths = [
        write_table3_csv(table3, SUMMARY_DIR / "Table3.csv"),
        write_table4_csv(table4, SUMMARY_DIR / "Table4.csv"),
    ]
    paper_paths = [
        write_table3_latex(table3, PAPER_FIGURES_DIR / "Table3.tex"),
        write_table4_latex(table4, PAPER_FIGURES_DIR / "Table4.tex"),
    ]
    return summary_paths, paper_paths


def render_titleless_paper_figures(experiment_dirs, all_lid_data):
    t10_experiment_dir = next(
        (path for path in experiment_dirs if path.name == "T10-2"),
        None,
    )
    if t10_experiment_dir is None:
        raise FileNotFoundError(
            "Paper Figures 2 and 3 require the T10-2 experiment."
        )

    df_lid = load_lid_dataframe(t10_experiment_dir)
    n_enrolments = get_n_enrolments(t10_experiment_dir, df_lid)
    metric_specs = [
        {
            "column": "p",
            "xlabel": r"Probability assigned to true identity, $p_{i,m_i}$",
            "baseline_val": 1.0 / n_enrolments,
            "baseline_label": f"Random Guess (p = 1/{n_enrolments})",
            "loc": "upper right",
            "destination": PAPER_FIGURES_DIR / "Figure2.pdf",
        },
        {
            "column": "LID",
            "xlabel": r"Local information disclosure, $\mathrm{LID}_i$ (bits)",
            "baseline_val": 0.0,
            "baseline_label": "No information disclosure",
            "loc": "upper left",
            "destination": PAPER_FIGURES_DIR / "Figure3.pdf",
        },
    ]

    exported_paths = []
    for spec in metric_specs:
        fig = create_metric_figure(
            df_lid,
            column=spec["column"],
            title=None,
            xlabel=spec["xlabel"],
            baseline_val=spec["baseline_val"],
            baseline_label=spec["baseline_label"],
            loc=spec["loc"],
        )
        fig.savefig(spec["destination"], format="pdf")
        plt.close(fig)
        exported_paths.append(spec["destination"])

    for show_legend, figure_name in [
        (True, "Figure4"),
        (False, "Figure4_no_legend"),
    ]:
        fig, _ = create_combined_lid_ccdf_figure(
            all_lid_data,
            title=None,
            show_legend=show_legend,
        )
        destination = PAPER_FIGURES_DIR / f"{figure_name}.pdf"
        fig.savefig(destination, format="pdf", bbox_inches="tight")
        plt.close(fig)
        exported_paths.append(destination)

    return exported_paths


def export_paper_figures(
    df_calibration,
    df_summary,
    experiment_dirs,
    all_lid_data,
):
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for figure_number in range(2, 6):
        (PAPER_FIGURES_DIR / f"Figure{figure_number}.png").unlink(missing_ok=True)
    for table_name in ["Table3", "Table4"]:
        (PAPER_FIGURES_DIR / f"{table_name}.csv").unlink(missing_ok=True)

    exported_paths = render_titleless_paper_figures(experiment_dirs, all_lid_data)
    skipped = []

    figure5_path = copy_figure(
        SUMMARY_DIR / "eer_vs_infodisc_scatter.pdf",
        "Figure5",
        PAPER_FIGURES_DIR,
        required=False,
    )
    if figure5_path:
        exported_paths.append(figure5_path)
    else:
        skipped.append("Figure5")

    figure5_no_legend_path = copy_figure(
        SUMMARY_DIR / "eer_vs_infodisc_scatter_no_legend.pdf",
        "Figure5_no_legend",
        PAPER_FIGURES_DIR,
        required=False,
    )
    if figure5_no_legend_path:
        exported_paths.append(figure5_no_legend_path)
    else:
        skipped.append("Figure5_no_legend")

    summary_table_paths, paper_table_paths = export_tables(df_calibration, df_summary)
    exported_paths.extend(paper_table_paths)

    return {
        "files": len(exported_paths),
        "skipped": skipped,
        "summary_table_paths": summary_table_paths,
    }


def export_readme_figures():
    README_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    t10_plots_dir = RESULTS_DIR / "T10-2" / "plots"
    sources = [
        (t10_plots_dir / "probability_distribution.png", "Figure2"),
        (t10_plots_dir / "lid_distribution.png", "Figure3"),
        (SUMMARY_DIR / "lid_combined_ccdf.png", "Figure4"),
    ]
    return [
        copy_figure(source_path, figure_name, README_FIGURES_DIR)
        for source_path, figure_name in sources
    ]


def plot_combined_summary(experiment_dirs):
    all_lid_data = {}
    for experiment_dir in experiment_dirs:
        df_lid = load_lid_dataframe(experiment_dir)
        all_lid_data[experiment_dir.name] = df_lid["LID"].to_numpy()

    summary_paths = []
    summary_paths.extend(plot_combined_lid_ccdf(all_lid_data, SUMMARY_DIR))
    df_summary, summary_table_path = build_summary_table(experiment_dirs, SUMMARY_DIR)
    summary_paths.append(summary_table_path)
    df_calibration, calibration_path = build_calibration_coeff_table(
        experiment_dirs, SUMMARY_DIR
    )
    summary_paths.append(calibration_path)
    eer_plot_paths = plot_summary_visuals(df_summary, SUMMARY_DIR)
    summary_paths.extend(eer_plot_paths)
    paper_export_summary = export_paper_figures(
        df_calibration,
        df_summary,
        experiment_dirs,
        all_lid_data,
    )
    summary_paths.extend(paper_export_summary["summary_table_paths"])
    readme_paths = export_readme_figures()

    return {
        "summary_table": df_summary,
        "summary_files": len(summary_paths),
        "eer_plot_created": bool(eer_plot_paths),
        "paper_files": paper_export_summary["files"],
        "paper_skipped": paper_export_summary["skipped"],
        "readme_files": len(readme_paths),
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
    print(f"Exported {summary['readme_files']} README figure files.")
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
    print("README figures can be found in results/readme_figures/.")
    print(SEPARATOR2)
    print()
