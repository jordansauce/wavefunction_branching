# %%
# IMPORTANT: You must run evolution_analysis/pickle_analysis.py before running this script
# This script plots the errors in the expectation values over different runs, as calculated in pickle_analysis.py

import copy
import glob
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogLocator
from natsort import natsorted
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm

NOW = datetime.now().strftime("%Y-%m-%d")

LINEWIDTH = 1.8

OP_NAMES_LATEX = {
    "〈σx〉": r"$\langle \sigma_x \rangle$",
    "〈σx σx〉": r"$\langle \sigma_x \sigma_x \rangle$",
    "〈σx 1 σx〉": r"$\langle \sigma_x I \sigma_x \rangle$",
    "〈σxA 2 σxB〉": r"$\langle \sigma_x II \sigma_x \rangle$",
    "〈σx 3 σx〉": r"$\langle \sigma_x III  \sigma_x \rangle$",
    "〈σxA 4 σxB〉": r"$\langle \sigma_x IIII \sigma_x \rangle$",
}

NICE_NAMES = {
    "None": "None",
    "do_nothing": "Do nothing",
    "truncation": "Truncation",
    "vertical_svd_micro_bsvd": "Vertical SVD",
    "pulling_through": "Pulling through",
    "bell_keep_classical": "Bell (keep classical)",
    "bell_discard_classical": "Bell (discard classical)",
    "bell_original_threshold_keep_classical": "Bell original threshold (keep classical)",
    "bell_original_threshold_discard_classical": "Bell original threshold (discard classical)",
    "rho_LM_MR_trace_norm_discard_classical_identical_blocks": "Local trace distance (discard classical)",
    "rho_LM_MR_trace_norm_identical_blocks": "Local trace distance (identical blocks)",
    "rho_LM_MR_trace_norm": "Local trace distance",
    "rho_half_LM_MR_trace_norm": "Local trace distance (rho_half)",
    "graddesc_global_reconstruction_non_interfering": "Global reconstruction \n\& block-diagonal non-interference",
    "graddesc_global_reconstruction_split_non_interfering": "Global reconstruction \n\& 2-site non-interference",
}

COLORS = {  # Gradient descent methods have colors
    "None": sns.color_palette("Paired")[0],
    "rho_LM_MR_trace_norm": sns.color_palette("Paired")[7],
    "rho_half_LM_MR_trace_norm": sns.color_palette("Paired")[6],
    "rho_LM_MR_trace_norm_discard_classical_identical_blocks": sns.color_palette("Paired")[4],
    "rho_LM_MR_trace_norm_identical_blocks": sns.color_palette("Paired")[5],
    "graddesc_global_reconstruction_non_interfering": sns.color_palette("Paired")[2],
    "graddesc_global_reconstruction_split_non_interfering": sns.color_palette("Paired")[3],
}

STYLES = {  # Iterative methods have styles
    "None": "solid",
    "truncation": "solid",
    "bell_keep_classical": "dotted",
    "bell_discard_classical": "dashed",
    "vertical_svd_micro_bsvd": "solid",
    # 'pulling_through': (0,(5,7)),
    "bell_original_threshold_keep_classical": (0, (5, 7)),
    "bell_original_threshold_discard_classical": (0, (7, 5)),
    "pulling_through": "dashdot",
}


OMIT_ITERATIVE_METHODS = [
    "bell_original_threshold_keep_classical",
    "bell_original_threshold_discard_classical",
]
OMIT_GRADDESC_METHODS = ["rho_half_LM_MR_trace_norm"]


def clean_method_name(method_name: str) -> str:
    method_name = method_name.replace("/", "")
    if method_name.startswith("f_"):
        method_name = method_name[2:]
    return method_name


OP_NAMES_FILENAMES = {
    "〈σx〉": "sigma_x",
    "〈σx σx〉": "sigma_x_sigma_x",
    "〈σx 1 σx〉": "sigma_x_I_sigma_x",
    "〈σxA 2 σxB〉": "sigma_x_II_sigma_x",
    "〈σx 3 σx〉": "sigma_x_III_sigma_x",
    "〈σxA 4 σxB〉": "sigma_x_IIII_sigma_x",
}


def get_method_name_iterative(filename):
    method = filename.split("-finite")[0]
    method_split = method.split("__")
    iterative_method = method_split[0].split("-")[-1]
    return clean_method_name(iterative_method)


def get_method_name_graddesc(filename):
    method = filename.split("-finite")[0]
    method_split = method.split("__")
    if len(method_split) > 1:
        graddesc_method = method_split[1]
    else:
        graddesc_method = "None"
    return clean_method_name(graddesc_method)


def get_color_from_row(row) -> tuple[float, float, float]:
    iterative_key = (
        "iterative_method" if "iterative_method" in row.keys() else "iterative method name"
    )
    graddesc_key = "graddesc_method" if "graddesc_method" in row.keys() else "graddesc method name"
    if row[iterative_key] == "truncation" or row[iterative_key] == "None":
        return (0.0, 0.0, 0.0)
    if row[graddesc_key] in COLORS:
        return COLORS[row[graddesc_key]]
    else:
        return (0.0, 0.0, 0.0)


def get_linewidth_from_row(row) -> float:
    iterative_key = (
        "iterative_method" if "iterative_method" in row.keys() else "iterative method name"
    )
    if row[iterative_key] == "truncation" or row[iterative_key] == "None":
        return LINEWIDTH * 2.5
    else:
        return LINEWIDTH


def get_color(method_name) -> tuple[float, float, float]:
    if method_name in COLORS:
        return COLORS[method_name]
    else:
        return (0.0, 0.0, 0.0)


def get_style(method_name) -> str:
    if method_name in STYLES:
        return STYLES[method_name]
    else:
        return "-"


def get_nice_name(method_name) -> str:
    if method_name in NICE_NAMES:
        return NICE_NAMES[method_name]
    else:
        return method_name


def preprocess_errors_data(df):
    df = df.assign(method=[x.split("-")[3] for x in df["filename"]])
    df.sort_values("method", inplace=True)

    # Filter for times before boundary effects have kicked in
    cutoff_times = {L: (0.25 * L - 2.0) for L in natsorted(df["L"].unique())}
    below_cutoff_time = [df.iloc[i]["time"] < cutoff_times[df.iloc[i]["L"]] for i in range(len(df))]
    df = df[below_cutoff_time]

    # Remove the bond dimension 200 runs
    df = df[df["max_bonds"] != 200]

    df = df[[x in [80, 128] for x in df["L"]]]

    # # Remove the runs not of the desired date
    # df = df[[('2025-02-19' in x) for x in df['filename']]]

    # Add a method_name column to the dataframe
    method_names_iterative = [get_method_name_iterative(x) for x in df["filename"]]
    method_names_graddesc = [get_method_name_graddesc(x) for x in df["filename"]]
    df = df.assign(**{"iterative method name": method_names_iterative})
    df = df.assign(**{"graddesc method name": method_names_graddesc})

    # Remove the methods we don't want to plot
    df = df[~df["iterative method name"].isin(OMIT_ITERATIVE_METHODS)]
    df = df[~df["graddesc method name"].isin(OMIT_GRADDESC_METHODS)]

    # Add a method name column to the dataframe by combining the iterative and gradient descent method names
    method_names = [
        iterative + " + " + graddesc
        for iterative, graddesc in zip(
            df["iterative method name"], df["graddesc method name"], strict=False
        )
    ]
    df = df.assign(**{"method name": method_names})
    df.sort_values("method name", inplace=True)

    # Add a LaTeX operator name column to the dataframe
    operators_latex = [OP_NAMES_LATEX[operator] for operator in df["operator"]]
    df = df.assign(**{"Operator": operators_latex})

    # Add columns for 'graddesc method name', 'iterative method name', 'color', 'style'
    df = df.assign(
        **{"nice graddesc method name": [get_nice_name(x) for x in df["graddesc method name"]]}
    )
    df = df.assign(
        **{"nice iterative method name": [get_nice_name(x) for x in df["iterative method name"]]}
    )
    df = df.assign(**{"color": [get_color_from_row(x) for _, x in df.iterrows()]})
    df = df.assign(**{"linewidth": [get_linewidth_from_row(x) for _, x in df.iterrows()]})
    df = df.assign(**{"style": [get_style(x) for x in df["iterative method name"]]})
    return df


def calculate_errors_relative_to_truncation(df):
    # Add columns for error_relative_to_truncation
    # First create groups based on the parameters that need to match
    groups = df.groupby(["L", "max_bonds", "operator", "time"])

    # Initialize the new column with NaN values
    df["error_relative_to_truncation"] = np.nan

    # For each group, calculate relative errors
    for name, group in tqdm(groups, desc="Calculating relative errors"):
        # Find the truncation row (where both method names are 'None')
        truncation_mask = (group["iterative method name"] == "None") & (
            group["graddesc method name"] == "None"
        )

        if not any(truncation_mask):
            continue

        # Take the average of the truncation rows
        truncation_error = group[truncation_mask]["expectation_value_error"].mean()

        # Calculate relative errors for all rows in this group
        group_indices = group.index
        df.loc[group_indices, "error_relative_to_truncation"] = (
            group["expectation_value_error"] / truncation_error
        )

    # Set truncation rows to 1.0
    truncation_mask = (df["iterative method name"] == "None") & (
        df["graddesc method name"] == "None"
    )
    df.loc[truncation_mask, "error_relative_to_truncation"] = 1.0

    return df


if __name__ == "__main__":
    # print('Loading the errors data')
    WORKSPACE_PATH = Path(__file__).parent.parent.absolute()
    errors_data_files = glob.glob(str(WORKSPACE_PATH / "evolution_analysis/data/*/errors.pkl"))
    errors_data_files = natsorted(errors_data_files)
    filename = errors_data_files[-1]
    # filename = "evolution_analysis/data/2025-02-13/errors.pkl"  #'evolution_analysis/data/2025-01-28/errors.pkl'
    print(f"loading from {filename}")
    errors_data = pd.read_pickle(filename)
    print("Loaded errors data.")

    print("Preprocessing errors data...")
    df = copy.deepcopy(errors_data)
    df = preprocess_errors_data(df)
    print("Preprocessed errors data.")

    # print("Calculating relative errors...")
    # df = calculate_errors_relative_to_truncation(df)
    # print("Calculated relative errors.")

    plots_prefix = filename.replace("/", "_").split(".")[0] + "_" + NOW
    # Plot the errors data
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = [8, 7]
    plt.rcParams["text.usetex"] = True

    # Add this configuration before any plotting code
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "font.size": 12,
        }
    )

    FONTSIZE_SMALL = 11
    FONTSIZE_LARGE = 15

    # %%
    # Plot the errors in each expectation value over time

    # df.keys() = (['filename', 'operator', 'time', 'L', 't_max', 'max_branches',
    #    'max_bonds', 'branch_at', 'expectation_value_analytic',
    #    'expectation_value_numeric', 'expectation_value_error', 'n_branches',
    #    'total_norm', 'expectation_value_standard_deviation', 'method',
    #    'average bond dimension', 'truncation error', 'norm',
    #    'average entanglement entropy', 'rho_trace_norm',
    #    'estimated_interference_error', 'tr(rho)', 'iterative method name',
    #    'graddesc method name', 'method name', 'Operator'],
    #   dtype='object')

    unique_max_bonds = natsorted(df["max_bonds"].unique())
    unique_Ls = natsorted(df["L"].unique())
    unique_branch_ats = natsorted(df["branch_at"].unique())
    unique_max_branches = natsorted(df["max_branches"].unique())
    unique_times = natsorted(df["time"].unique())
    unique_t_maxs = natsorted(df["t_max"].unique())
    unique_operators = natsorted(df["operator"].unique())

    operator = "〈σx〉"  #'〈σxA 4 σxB〉'
    L = unique_Ls[0]
    max_bonds = unique_max_bonds[0]
    max_branches = unique_max_branches[0]

    # Divide the operators by the operator from truncation at the same time

    print(f"L = {L}")
    print(f"max_bonds = {max_bonds}")
    print(f"max_branches = {max_branches}")
    print(f"operator = {operator}")
    _df = copy.deepcopy(df)
    _df = _df[_df["L"] == L]
    _df = _df[_df["max_bonds"] == max_bonds]
    _df = _df[_df["max_branches"] == max_branches]
    _df = _df[_df["operator"] == operator]

    # Create figure and axes grid based on number of combinations
    n_rows = len(unique_Ls)
    n_cols = len(unique_max_bonds)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), dpi=180, sharex=True, sharey=True
    )

    for i, L in tqdm(enumerate(unique_Ls)):
        for j, max_bonds in enumerate(unique_max_bonds):
            # print(f'L = {L}')
            # print(f'max_bonds = {max_bonds}')
            # print(f'max_branches = {max_branches}')
            # print(f'operator = {operator}')
            _df = copy.deepcopy(df)
            _df = _df[_df["L"] == L]
            _df = _df[_df["max_bonds"] == max_bonds]
            _df = _df[_df["max_branches"] == max_branches]
            _df = _df[_df["operator"] == operator]

            # Get current axis based on position
            ax = (
                axes[i, j]
                if n_rows > 1 and n_cols > 1
                else axes[i]
                if n_rows > 1
                else axes[j]
                if n_cols > 1
                else axes
            )

            filenames = natsorted(_df["filename"].unique())
            for filename in filenames:
                _df_filename = _df[_df["filename"] == filename]
                graddesc_method_name = _df_filename["graddesc method name"].unique()
                assert len(graddesc_method_name) == 1
                graddesc_method_name = clean_method_name(graddesc_method_name[0])
                iterative_method_name = _df_filename["iterative method name"].unique()
                assert len(iterative_method_name) == 1
                iterative_method_name = clean_method_name(iterative_method_name[0])
                sort_args = np.argsort(_df_filename["time"])
                style = _df_filename["style"].unique()[0]
                color = _df_filename["color"].unique()[0]
                linewidth = _df_filename["linewidth"].unique()[0]
                y = np.array(_df_filename["expectation_value_error"])[sort_args]
                y_smooth = gaussian_filter1d(y, sigma=20)
                # Set z-order based on iterative method - 'None' should be beneath others
                z_order = 1 if iterative_method_name == "None" else 2
                ax.plot(
                    np.array(_df_filename["time"])[sort_args],
                    y_smooth,
                    color=color,
                    linestyle=style,
                    linewidth=linewidth,
                    zorder=z_order,
                )

            ax.set_yscale("log")
            ax.set_ylim(1e-4, 2e-2)
            # Add minor ticks to log scale
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))  # type: ignore
            ax.grid(True, which="minor", alpha=0.2)
            ax.grid(True, which="major", alpha=0.5)
            ax.set_title(f"L={L}, max_bonds={max_bonds}", fontsize=FONTSIZE_SMALL)
    plt.tight_layout()

    # Create separate legends for iterative and gradient descent methods
    # First create dummy lines for the iterative methods legend
    active_iterative_methods = _df["iterative method name"].unique()
    iterative_legend_elements = [
        plt.Line2D(  # type: ignore
            [0],
            [0],
            color="black",
            linestyle=style,
            label=NICE_NAMES[method],
        )
        for method, style in STYLES.items()
        if method != "None" and method in active_iterative_methods
    ]

    # Create dummy patches for the gradient descent methods legend
    active_graddesc_methods = _df["graddesc method name"].unique()
    graddesc_legend_elements = [
        plt.Line2D(  # type: ignore
            [0],
            [0],
            color=color,
            linestyle="-",
            label=NICE_NAMES[method],
        )
        for method, color in COLORS.items()
        if method in active_graddesc_methods
    ]

    # Add both legends
    fig.legend(
        handles=iterative_legend_elements,
        title="Iterative Methods",
        bbox_to_anchor=(1.01, 0.7),
        loc="center left",
        borderaxespad=0,
    )
    fig.legend(
        handles=graddesc_legend_elements,
        title="Gradient Descent Methods",
        bbox_to_anchor=(1.01, 0.3),
        loc="center left",
        borderaxespad=0,
    )

    fig.supylabel(
        f"Error in {OP_NAMES_LATEX[operator]} (smoothed)", fontsize=FONTSIZE_LARGE, x=-0.02
    )
    fig.supxlabel("Time", fontsize=FONTSIZE_LARGE, y=-0.02)
    plt.savefig(
        WORKSPACE_PATH
        / f"evolution_analysis/plots/{filename.replace('/', '_').split('.')[0]}_{NOW}_error_over_time_{OP_NAMES_FILENAMES[operator]}.pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        WORKSPACE_PATH
        / f"evolution_analysis/plots/{filename.replace('/', '_').split('.')[0]}_{NOW}_error_over_time_{OP_NAMES_FILENAMES[operator]}.png",
        bbox_inches="tight",
        dpi=180,
    )
    plt.show()

    # %%
    # Plot a bar graph of the average error in each method
    def plot_error_bars(
        df, operator, padding=0.2, show_errorbars=True, relative_to_truncation=True, log_scale=True
    ):
        """
        Plot bar charts comparing different methods' performance relative to truncation.

        Args:
            df: DataFrame containing the analysis data
            operator: Which operator to analyze
            padding: Spacing between groups of bars
            show_errorbars: Whether to show error bars on the plots
        """
        if relative_to_truncation:
            to_plot = "error_relative_to_truncation"
        else:
            to_plot = "expectation_value_error"

        # Filter for the specified operator
        _df = df[df["operator"] == operator].copy()
        _df = _df[_df["time"] >= 10.0]

        # Get unique methods
        iterative_methods = sorted([m for m in _df["iterative method name"].unique()])

        # Setup figure
        width_ratios = [
            (
                len(
                    _df[_df["iterative method name"] == iterative_method][
                        "graddesc method name"
                    ].unique()
                )
                + 2 * padding
            )
            for iterative_method in iterative_methods
        ]
        fig, axes = plt.subplots(
            1,
            len(iterative_methods),
            figsize=(13, 5),
            dpi=180,
            sharey=True,
            width_ratios=width_ratios,
            gridspec_kw={"wspace": 0},
        )
        if len(iterative_methods) == 1:
            axes = [axes]
        all_graddesc_methods = natsorted(_df["graddesc method name"].unique())
        # Plot each iterative method in its own subplot
        for i, iterative_method in enumerate(iterative_methods):
            method_df = _df[_df["iterative method name"] == iterative_method]

            # Calculate statistics for each gradient descent method
            graddesc_methods = natsorted(method_df["graddesc method name"].unique())
            print(graddesc_methods)
            stats = {}

            for method in graddesc_methods:
                method_data = method_df[method_df["graddesc method name"] == method][to_plot]

                # # Bootstrap confidence intervals
                # n_bootstrap = 1000
                # bootstrap_means = np.zeros(n_bootstrap)
                # for j in range(n_bootstrap):
                #     bootstrap_sample = np.random.choice(method_data, size=len(method_data), replace=True)
                #     bootstrap_means[j] = np.mean(bootstrap_sample)
                # ci = np.percentile(bootstrap_means, [2.5, 97.5])

                stats[method] = {
                    "mean": np.mean(method_data),
                    # 'ci': ci
                }

            # Plot bars
            x_pos = np.arange(len(graddesc_methods))
            means = [stats[m]["mean"] for m in graddesc_methods]
            # cis = np.array([stats[m]['ci'] for m in graddesc_methods])
            # yerr = np.abs(cis - np.array(means)[:, np.newaxis]).T if show_errorbars else None

            # Plot bars with colors from COLORS dictionary
            axes[i].bar(
                x_pos,
                means,
                # yerr=yerr,
                width=1.0,
                color=[COLORS[m] for m in graddesc_methods],
                alpha=1.0,
                edgecolor="white",
                capsize=5,
                error_kw={"alpha": 0.7, "ecolor": "black", "linewidth": 2.0},
            )

            # Show individual runs
            for j, method in enumerate(graddesc_methods):
                _method_df = method_df[method_df["graddesc method name"] == method]
                unique_max_bonds = natsorted(_method_df["max_bonds"].unique())
                unique_Ls = natsorted(_method_df["L"].unique())
                t_max = max(_method_df["time"])
                for max_bonds in unique_max_bonds:
                    for L in unique_Ls:
                        _df_mb = _method_df[_method_df["max_bonds"] == max_bonds]
                        _df_mb = _df_mb[_df_mb["L"] == L]
                        if len(_df_mb) == 0:
                            continue
                        sort_args = np.argsort(_df_mb["time"])
                        ts_sorted = np.array(_df_mb["time"])[sort_args]
                        errors_sorted = np.array(_df_mb[to_plot])[sort_args]
                        ts_normalized = (ts_sorted - ts_sorted[0]) / (t_max - ts_sorted[0])
                        errors_smoothed = gaussian_filter1d(errors_sorted, sigma=20)
                        axes[i].plot(
                            j - 0.4 + 0.8 * ts_normalized,
                            errors_smoothed,
                            alpha=0.2,
                            color="black",
                            linewidth=0.5,
                        )

            # Customize subplot
            truncation_error = np.mean(
                _df[_df["graddesc method name"] == "None"][_df["iterative method name"] == "None"][
                    to_plot
                ]
            )
            axes[i].axhline(y=truncation_error, color="black", linestyle="--", linewidth=0.8)
            axes[i].set_xticks([])
            axes[i].set_title(NICE_NAMES[iterative_method], fontsize=FONTSIZE_SMALL, y=-0.09)

            xlim = axes[i].get_xlim()
            axes[i].set_xlim(xlim[0] - padding, xlim[1] + padding)
            # if i == 0:
            #     axes[i].set_xlim(xlim[0] - padding, xlim[1] + 3 * padding)

            # Remove unnecessary spines
            axes[i].spines["top"].set_visible(False)
            axes[i].spines["right"].set_visible(False)
            axes[i].spines["bottom"].set_visible(False)
            # axes[i].spines['left'].set_visible(False)

            if log_scale:
                axes[i].set_yscale("log")
                axes[i].yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))  # type: ignore
                axes[i].grid(True, which="minor", alpha=0.25)
                axes[i].grid(True, which="major", alpha=1)

        # Global figure customization
        axes[0].set_ylabel(
            f"Error {'relative to truncation' * relative_to_truncation} in {OP_NAMES_LATEX[operator]}",
            fontsize=FONTSIZE_LARGE,
        )

        plt.suptitle("Iterative method", y=0.03, fontsize=FONTSIZE_LARGE)

        # Create legend
        legend_elements = [
            plt.Rectangle(  # type: ignore
                (0, 0),
                1,
                1,
                facecolor=COLORS[method],
                label=NICE_NAMES[method],
            )
            for method in all_graddesc_methods
        ]

        fig.legend(
            handles=legend_elements,
            title="Gradient descent method",
            bbox_to_anchor=(0.5, -0.05),
            loc="upper center",
            ncol=len(legend_elements) // 2 + len(legend_elements) % 2,
            fontsize=FONTSIZE_SMALL,
            title_fontsize=FONTSIZE_LARGE,
        )

        plt.tight_layout()
        return fig, axes

    # Example usage:
    # operator = '〈σx〉'
    for relative_to_truncation in [False, True]:
        for operator in ["〈σx〉", "〈σx σx〉", "〈σxA 4 σxB〉"]:
            fig, axes = plot_error_bars(df, operator, relative_to_truncation=relative_to_truncation)
            plt.savefig(
                WORKSPACE_PATH
                / f"evolution_analysis/plots/{NOW}_error_bars_{OP_NAMES_FILENAMES[operator]}_{'relative_to_truncation' if relative_to_truncation else 'absolute'}.pdf",
                bbox_inches="tight",
            )
            plt.savefig(
                WORKSPACE_PATH
                / f"evolution_analysis/plots/{NOW}_error_bars_{OP_NAMES_FILENAMES[operator]}_{'relative_to_truncation' if relative_to_truncation else 'absolute'}.png",
                bbox_inches="tight",
                dpi=180,
            )
            plt.show()
    # %%
    # Try to anayze how much each of these things contribute to the final expectation value error:
    #     Instantaneous interference errors
    #     Instantaneous global reconstruction errors
    #     Instantaneous local reconstruction errors

    # ['filename', 'operator', 'time', 'L', 't_max', 'max_branches',
    #    'max_bonds', 'branch_at', 'expectation_value_analytic',
    #    'expectation_value_numeric', 'expectation_value_error', 'n_branches',
    #    'total_norm', 'expectation_value_standard_deviation', 'method',
    #    'rho_trace_norm', 'estimated_interference_error',
    #    'average entanglement entropy', 'norm', 'average bond dimension',
    #    'truncation error', 'tr(rho)', 'iterative method name',
    #    'graddesc method name', 'method name', 'Operator',
    #    'nice graddesc method name', 'nice iterative method name', 'color',
    #    'style', 'error_relative_to_truncation']

    # Independent variables:
    #     estimated_interference_error
    #     truncation error
    #     tr(rho)
    #     total_norm
    #     expectation_value_standard_deviation

    # Dependent variables:
    #     expectation_value_error

    from sklearn.linear_model import LinearRegression

    operator = "〈σx〉"
    df_op = df[df["operator"] == operator]
    df_op = df_op[df_op["time"] >= 5.0]
    df_op = df_op[df_op["L"] == 80]

    # Attempt to estimate the contribution of each of these variables to the final expectation value error
    for independent_variable in [
        "estimated_interference_error",
        "truncation error",
        "tr(rho)",
        "total_norm",
        "expectation_value_standard_deviation",
    ]:
        # Fit a linear model to the data
        iv_values = df_op[independent_variable].values
        X = np.log10(abs(iv_values.reshape(-1, 1)) + 1e-15)
        y = np.log10(abs(df_op["expectation_value_error"].values) + 1e-15)
        model = LinearRegression()
        model.fit(X, y)
        print(f"model.coef_ = {model.coef_}")
        print(f"model.intercept_ = {model.intercept_}")
        sns.scatterplot(
            data=df_op,
            x=independent_variable,
            y="expectation_value_error",
            hue="graddesc method name",
            style="iterative method name",
        )
        X_range = np.linspace(np.min(X), np.max(X), 1000).reshape(-1, 1)
        plt.plot(10**X_range, 10 ** model.predict(X_range), color="black", linewidth=0.5)
        # plt.yscale('log')
        # plt.xscale('log')
        plt.show()

    # %%

    print("Plotting a scatterplot of n_branches vs operator error")
    for operator in natsorted(df["operator"].unique())[:1]:
        for error in [
            "one_minus_overlap_pure",
            "trace_distance_LM",
            "expectation_value_standard_deviation",
            "expectation_value_error",
            "truncation error",
        ]:
            if operator in OP_NAMES_LATEX:
                operator_latex = OP_NAMES_LATEX[operator]
            else:
                operator_latex = (
                    operator.replace("x", "$_x$")
                    .replace("〈", r"$\langle$")
                    .replace("〉", r"$\rangle$")
                    .replace("σ", r"$\sigma$")
                )
            df_op = df[df["operator"] == operator]
            plt.figure(dpi=180)
            g = sns.relplot(
                data=df_op,
                y=error,
                x="time",
                hue="graddesc method name",
                col="iterative method name",
                row="L",
                kind="line",
                alpha=1.0,
            )
            # plt.yscale('log')
            # plt.ylim(1e-4,1.0)
            plt.show()
        # df_op = df[df['operator'] == operator]
        # plt.figure(dpi=180)
        # g = sns.relplot(data=df_op, y='truncation error', x='time', hue='graddesc method name', col='iterative method name', row='L', kind='line', alpha=1.0)
        # plt.yscale('log')
        # plt.ylim(1e-4,1.0)
        # plt.show()

    # %%
    print("Plotting the errors in each expectation value over time")
    for operator in ["〈σx〉"]:  # natsorted(df['operator'].unique()):
        if operator in OP_NAMES_LATEX:
            operator_latex = OP_NAMES_LATEX[operator]
        else:
            operator_latex = (
                operator.replace("x", "$_x$")
                .replace("〈", r"$\langle$")
                .replace("〉", r"$\rangle$")
                .replace("σ", r"$\sigma$")
            )
        df_op = df[df["operator"] == operator]
        operator_safe = (
            operator.replace("〈", "").replace("〉", "").replace("σ", "sigma_").replace(" ", "_")
        )

        plt.figure(dpi=320)
        g = sns.relplot(
            data=df_op,
            x="time",
            y="expectation_value_error",
            col="iterative method name",
            row="L",
            hue="graddesc method name",
            style="max_bonds",
            alpha=0.7,
            kind="line",
        )  # , palette=sns.color_palette("Set2"))
        # g2 = sns.relplot(data=df_op, x='time', y=np.minimum((1.0 - df_op['total_norm']), df_op['expectation_value_standard_deviation']/np.sqrt(df_op['max_branches'])), alpha=0.5, hue='method',col="max_branches", row="L", legend=False, kind='line', linestyle='--')
        g.fig.suptitle("Errors in " + operator_latex, fontsize=20)
        g.fig.subplots_adjust(top=0.9)
        # plt.yscale('log')
        # plt.tight_layout()
        plt.savefig(
            WORKSPACE_PATH / f"evolution_analysis/plots/{NOW}_{operator_safe}_errors_over_time.png"
        )
        plt.savefig(
            WORKSPACE_PATH / f"evolution_analysis/plots/{NOW}_{operator_safe}_errors_over_time.pdf"
        )
        plt.show()

    # %%
    print("Plotting the errors in each expectation value over time, for fixed max_branches")
    for operator in natsorted(df["operator"].unique()):
        if operator in OP_NAMES_LATEX:
            operator_latex = OP_NAMES_LATEX[operator]
        else:
            operator_latex = (
                operator.replace("x", "$_x$")
                .replace("〈", r"$\langle$")
                .replace("〉", r"$\rangle$")
                .replace("σ", r"$\sigma$")
            )
        df_op = df[df["operator"] == operator]
        operator_safe = (
            operator.replace("〈", "").replace("〉", "").replace("σ", "sigma_").replace(" ", "_")
        )
        for max_branches in natsorted(df_op["max_branches"].unique()):
            if int(max_branches) > 1:
                plot_now = [
                    x["max_branches"] == max_branches or "Truncation" in x["method name"]
                    for i, x in df_op.iterrows()
                ]
                df_op_mb = df_op[plot_now]

                plt.rcParams["figure.figsize"] = [12, 10]
                plt.figure(dpi=180)
                sns.lineplot(
                    data=df_op_mb,
                    x="time",
                    y="expectation_value_error",
                    hue="method name",
                    alpha=0.85,
                )
                plt.ylabel("Errors in " + operator_latex, fontsize=12)
                plt.title(f"Max branches = {max_branches}")
                plt.tight_layout()
                plt.savefig(
                    f"evolution_analysis/plots/{NOW}_{operator_safe}_errors_over_time_max_branches_{max_branches}.png"
                )
                plt.savefig(
                    f"evolution_analysis/plots/{NOW}_{operator_safe}_errors_over_time_max_branches_{max_branches}.pdf"
                )
                plt.show()

    print("#################################################################################")
    print("#################################################################################")
    print("\n\n\n FINISHED error_analysis.py \n\n")
    print("#################################################################################")
    print("#################################################################################")
