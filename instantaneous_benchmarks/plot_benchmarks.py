"""
This script plots benchmark results for simultaneous block diagonalization & SVD algorithms
These algorithms take a set of matrices {A_1 ... A_N} and perform similarity transformations
to yield a set of block diagonal matrices {B_1 ... B_n} with the same finest block structure.

THIS IS ONLY A PLOTTING SCRIPT, NOT A BENCHMARK SCRIPT.

This script depends on benchmark_results.json file with the metrics of each of the
block-diagonalization methods on each of the test inputs. To generate the benchmark_results.json
file, use the generate_benchmark_results.py script.
That script in turn depends on input matrices and a block_diagonal_test_data/directory.json file,
so start by generating these using generate_test_inputs.py if you don't have them already.
"""

# %%
import copy
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from natsort import natsorted

from evolution_analysis.error_analysis import (
    COLORS,
    LINEWIDTH,
    NICE_NAMES,
    STYLES,
    get_color,
    get_color_from_row,
    get_linewidth_from_row,
    get_nice_name,
    get_style,
)

PROP_NICE_NAMES = {
    "global_reconstruction_error_trace_distance": "Global reconstruction error",
    "global_reconstruction_error_trace_distance_pure": "Global reconstruction error",
    "interference_error_trace_distance": "Interference error",
    "reconstruction_plus_interference_error_trace_distance": "Reconstruction + Interference error",
    "costFun_LM_MR_trace_distance": "Local trace distance (LM MR)",
    "costFun_split_trace_distance": "Local trace distance (2-site)",
    "one_minus_overlap_pure": "1 - Overlap (pure)",
    "costFun_LM_MR_rho_half_frobenius": "Local trace distance (rho_half LM MR)",
    "norms_geometric_mean": "Norms geometric mean",
}


if __name__ == "__main__":
    current_path = Path(__file__).parent.absolute()
    # current_path = Path("instantaneous_benchmarks")
    results_paths = natsorted(
        glob.glob(str(current_path / "benchmark_results/benchmark_results*.json"))
    )
    assert len(results_paths) > 0, "No results found at " + str(current_path / "benchmark_results")
    results_path = results_paths[-1]

    # results_path = "instantaneous_benchmarks/benchmark_results/benchmark_results_2025-02-18_06-28-41--ising.json"
    # results_path = "instantaneous_benchmarks/benchmark_results/benchmark_results_2025-02-17_04-48-29--ising.json"
    # results_path = "instantaneous_benchmarks/benchmark_results/benchmark_results_2025-01-21_16-40-38--ising.json"
    # results_path = "instantaneous_benchmarks/benchmark_results/benchmark_results_2025-01-21_11-59-05--random-quench.json"
    # results_path = natsorted(glob.glob(str(current_path / 'benchmark_results/benchmark_results_2024-10-24_22-12-57.json')))[-1]
    print(f"results_path for generating plots: {results_path}")
    results = json.load(open(results_path, "rb"))
    n_entries = np.median([len(results[key]) for key in results])
    skipping = {key: len(results[key]) for key in results if len(results[key]) != n_entries}
    print(f"Skipping these columns as n_entries!={n_entries}: {skipping}")
    results = {key: results[key] for key in results if len(results[key]) == n_entries}
    results = pd.DataFrame(results)

    save_folder_name = Path(results_path).name.split("benchmark_results_")[-1].split(".json")[0]
    plots_folder = current_path / f"plots/{save_folder_name}"
    if not plots_folder.exists():
        plots_folder.mkdir(parents=True, exist_ok=True)
    print(f"plots_folder for saving plots: {plots_folder}")

    # Lists are not hashable for plots so make a new column of gauge forms as strings:
    results_dict = dict(results)
    results_dict["form_str"] = [str(results["form"][i]) for i in range(len(results))]
    results = pd.DataFrame(results_dict)
    # sort the results
    results = results.iloc[
        results["method_name"]
        .replace("do_nothing", "aaa")
        .replace(
            "bell_decomp_iterative_keep_classical_plus_graddesc_block_diagonal_rho_half", "xxx"
        )
        .replace("truncation", "aab")
        .replace("truncation_two_thirds", "aac")
        .replace(
            "bell_decomp_orig_graddesc_discard_classical",
            "bell_decomp_iterative_discard_classical_plus_graddesc",
        )
        .replace(
            "bell_decomp_orig_graddesc_keep_classical",
            "bell_decomp_iterative_keep_classical_plus_graddesc",
        )
        .replace(
            "iterative_pulling_through_no_jumps_z0_dim_three_quarters",
            "iterative_pulling_through_z0_dim_three_quarters_no_jumps",
        )
        .replace(
            "iterative_pulling_through_no_jumps_z0_dim_two_thirds",
            "iterative_pulling_through_z0_dim_two_thirds_no_jumps",
        )
        .to_numpy()
        .argsort()
    ]

    palette = list(sns.color_palette("Paired"))
    method_names = results["method_name"].unique()
    print(method_names)

    if "bell_decomp_iterative_discard_classical_plus_graddesc_rho_half" in method_names:
        ind = np.argwhere(
            method_names == "bell_decomp_iterative_discard_classical_plus_graddesc_rho_half"
        ).item()
        c = np.average(np.array([palette[ind - 2], palette[ind - 1]]), axis=0)
        palette.insert(ind, c)

    if "bell_decomp_iterative_keep_classical_plus_graddesc_rho_half" in method_names:
        ind = np.argwhere(
            method_names == "bell_decomp_iterative_keep_classical_plus_graddesc_rho_half"
        ).item()
        c = np.average(np.array([palette[ind - 2], palette[ind - 1]]), axis=0)
        palette.insert(ind, c)

    if "bell_decomp_iterative_keep_classical_plus_graddesc_block_diagonal_rho_half" in method_names:
        ind = np.argwhere(
            method_names
            == "bell_decomp_iterative_keep_classical_plus_graddesc_block_diagonal_rho_half"
        ).item()
        palette.pop(ind)
    # palette

    show_errorbars = "ising" not in results_path

    # Add these imports near the top with other imports

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

    results = results.assign(
        **{"nice graddesc method name": [get_nice_name(x) for x in results["graddesc_method"]]}
    )
    results = results.assign(
        **{"nice iterative method name": [get_nice_name(x) for x in results["iterative_method"]]}
    )
    results = results.assign(**{"color": [get_color_from_row(x) for _, x in results.iterrows()]})
    results = results.assign(
        **{"linewidth": [get_linewidth_from_row(x) for _, x in results.iterrows()]}
    )
    results = results.assign(**{"style": [get_style(x) for x in results["iterative_method"]]})

    # Filter out rows that don't have a corresponding truncation row
    for method in results["method_name"].unique():
        if method != "truncation":
            method_data = results[results["method_name"] == method]
            for i, row in method_data.iterrows():
                timestamp = row["timestamp"]
                df_now = results[results["timestamp"] == timestamp]
                truncation = df_now[df_now["method_name"] == "truncation"]
                if len(truncation) == 0:
                    results = results.drop(i)
                    print(
                        f"Dropping {method} at timestamp {timestamp} because no truncation row found"
                    )

    results = results[results["chi_max"] < 400]

    results["overlap_pure"] = 1.0 - results["one_minus_overlap_pure"]
    results["global_reconstruction_error_trace_distance"] = np.sqrt(
        np.abs(1.0 - np.abs(results["overlap_pure"]) ** 2)
    )
    results["global_reconstruction_error_trace_distance_pure"] = np.sqrt(
        np.abs(1.0 - np.abs(results["overlap_pure"]) ** 2)
    )

    # results['global_reconstruction_error_trace_distance'] = results['one_minus_overlap_pure']**0.5

    # res

    # %% Plot instantaneous errors over time

    show_do_nothing = False
    FONTSIZE_LARGE = 16
    FONTSIZE_HUGE = 22

    _df = copy.deepcopy(results)

    omit_methods = [
        "bell_keep_classical__graddesc_global_reconstruction_split_non_interfering",
        "bell_keep_classical__rho_LM_MR_trace_norm",
        "bell_keep_classical__rho_half_LM_MR_trace_norm",
        "bell_keep_classical_old__rho_LM_MR_trace_norm_identical_blocks_old",
        "bell_discard_classical_old__rho_LM_MR_trace_norm_discard_classical_identical_blocks_old",
        "bell_keep_classical__rho_LM_MR_trace_norm_identical_blocks_old",
        "bell_discard_classical__rho_LM_MR_trace_norm_discard_classical_identical_blocks_old",
        "bell_discard_classical_old",
        "bell_keep_classical_old",
    ]

    if not show_do_nothing:
        omit_methods.append("do_nothing")
    _df = _df.reset_index(drop=True)

    _df = _df[[x not in omit_methods for x in _df["method_name"]]]

    _df = _df[_df["t"] <= 7]

    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))

    props = [
        "costFun_LM_MR_trace_distance",
        "global_reconstruction_error_trace_distance_pure",
        "interference_error_trace_distance",
    ]

    chi_maxes = [50, 100, 200]

    for i, prop in enumerate(props):
        for j, chi_max in enumerate(chi_maxes):
            ax = axes[i, j]

            df_chi = _df[_df["chi_max"] == chi_max]
            xprop = "t"
            normalize_to_truncation = False

            # Create line plot
            for method in df_chi["method_name"].unique():
                method_data = df_chi[df_chi["method_name"] == method]
                method_data = method_data.sort_values(by=xprop)
                normalized_data = copy.deepcopy(method_data[prop])
                if normalize_to_truncation:
                    for idx, row in method_data.iterrows():
                        timestamp = row["timestamp"]
                        df_now = df_chi[df_chi["timestamp"] == timestamp]
                        df_truncation_now = df_now[df_now["method_name"] == "truncation"]
                        assert len(df_truncation_now) == 1, f"{len(df_truncation_now)} != 1"
                        truncation_prop = df_truncation_now[prop].item()
                        normalized_data.loc[idx] = row[prop] / (abs(truncation_prop) + 1e-10)

                sns.lineplot(
                    x=np.array(method_data[xprop]),
                    y=np.array(normalized_data),
                    color=method_data["color"].iloc[0],
                    linestyle=method_data["style"].iloc[0],
                    linewidth=method_data["linewidth"].iloc[0],
                    ax=ax,
                )

            ax.set_yscale("log")
            if prop == "interference_error_trace_distance":
                ax.set_yscale("symlog", linthresh=1e-3)

            # Only show x-label on bottom row
            if i == 2:
                ax.set_xlabel("Time", fontsize=FONTSIZE_LARGE)

            # Only show y-label on leftmost column
            if j == 0:
                if prop == "interference_error_trace_distance":
                    nice_name = "Interference"
                elif prop == "global_reconstruction_error_trace_distance_pure":
                    nice_name = "Global"
                elif prop == "costFun_LM_MR_trace_distance":
                    nice_name = "Local"
                ax.set_ylabel(f"{nice_name}", fontsize=FONTSIZE_LARGE)

            # Add chi_max as title for top row
            if i == 0:
                ax.set_title(f"{chi_max}", fontsize=FONTSIZE_LARGE)
    # Larger y axis suptitle
    fig.supylabel("Instantaneous error", fontsize=FONTSIZE_HUGE)
    fig.suptitle("Maximum bond dimension", fontsize=FONTSIZE_HUGE, x=0.55)

    # Create legends outside the subplots
    iterative_legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle=style,
            linewidth=_df[_df["iterative_method"] == method].iloc[0]["linewidth"],
            label=NICE_NAMES[method],
        )
        for method, style in STYLES.items()
        if method != "None"
    ]

    graddesc_legend_elements = [
        plt.Line2D(
            [0], [0], color=color, linestyle="-", linewidth=LINEWIDTH, label=NICE_NAMES[method]
        )
        for method, color in COLORS.items()
    ]

    # Add legends to the right of the subplots
    fig.legend(
        handles=iterative_legend_elements,
        title="Iterative Methods",
        bbox_to_anchor=(1.002, 0.7),
        loc="center left",
        borderaxespad=0,
        fontsize=FONTSIZE_LARGE,
        title_fontsize=FONTSIZE_HUGE,
    )
    fig.legend(
        handles=graddesc_legend_elements,
        title="Gradient Descent Methods",
        bbox_to_anchor=(1.002, 0.35),
        loc="center left",
        borderaxespad=0,
        fontsize=FONTSIZE_LARGE,
        title_fontsize=FONTSIZE_HUGE,
    )

    plt.tight_layout()

    plt.savefig(plots_folder / "instantaneous_errors_grid.png", dpi=300, bbox_inches="tight")
    plt.savefig(plots_folder / "instantaneous_errors_grid.pdf", bbox_inches="tight")
    plt.show()
    print(f"Saved instantaneous errors grid to {plots_folder / 'instantaneous_errors_grid.png'}")

    # %% Plot instantaneous errors over time

    show_do_nothing = False

    _df = copy.deepcopy(results)

    omit_methods = [
        "bell_keep_classical__graddesc_global_reconstruction_split_non_interfering",
        "bell_keep_classical__rho_LM_MR_trace_norm",
        "bell_keep_classical__rho_half_LM_MR_trace_norm",
        "bell_keep_classical_old__rho_LM_MR_trace_norm_identical_blocks_old",
        "bell_discard_classical_old__rho_LM_MR_trace_norm_discard_classical_identical_blocks_old",
        "bell_keep_classical__rho_LM_MR_trace_norm_identical_blocks_old",
        "bell_discard_classical__rho_LM_MR_trace_norm_discard_classical_identical_blocks_old",
        "bell_discard_classical_old",
        "bell_keep_classical_old",
    ]

    if not show_do_nothing:
        omit_methods.append("do_nothing")
    _df = _df.reset_index(drop=True)

    _df = _df[[x not in omit_methods for x in _df["method_name"]]]

    # prop = 'global_reconstruction_error_trace_distance'
    # _df = _df[_df['probs_geometric_mean'] > 0.001]
    # _df = _df[np.logical_or(_df['probs_geometric_mean'] > 0.001, _df['method_name'] == "truncation")]

    _df = _df[_df["t"] <= 7]

    chi_max = 200
    _df = _df[_df["chi_max"] == chi_max]
    for prop in [
        # 'global_reconstruction_error_trace_distance_pure',
        # 'interference_error_trace_distance',
        # 'one_minus_overlap_pure',
        "costFun_LM_MR_trace_distance",
        # 'costFun_split_trace_distance',
        # 'costFun_LM_MR_rho_half_frobenius'
    ]:
        xprop = "t"  #'second-dominant TM eigenvalue magnitude'
        # normalize_to_truncation = prop != 'interference_error_trace_distance'
        normalize_to_truncation = False

        fig = plt.figure()

        # for method in _df['method_name'].unique():
        #     method_data = _df[_df['method_name'] == method]
        # Create scatter plot
        for method in _df["method_name"].unique():
            method_data = _df[_df["method_name"] == method]
            method_data = method_data.sort_values(by=xprop)
            normalized_data = copy.deepcopy(method_data[prop])
            if normalize_to_truncation:
                for i, row in method_data.iterrows():
                    timestamp = row["timestamp"]
                    df_now = _df[_df["timestamp"] == timestamp]
                    df_truncation_now = df_now[df_now["method_name"] == "truncation"]
                    assert len(df_truncation_now) == 1, f"{len(df_truncation_now)} != 1"
                    truncation_prop = df_truncation_now[prop].item()
                    normalized_data.loc[i] = row[prop] / (abs(truncation_prop) + 1e-10)
            # plt.plot(
            #     np.array(method_data[xprop]),
            #     np.array(normalized_data),
            #     color=method_data['color'].iloc[0],
            #     linestyle=method_data['style'].iloc[0],
            #     linewidth=linewidth,
            # )
            sns.lineplot(
                x=np.array(method_data[xprop]),
                y=np.array(normalized_data),
                color=method_data["color"].iloc[0],
                linestyle=method_data["style"].iloc[0],
                linewidth=method_data["linewidth"].iloc[0],
            )
        plt.yscale("log")
        if prop == "interference_error_trace_distance":
            # plt.ylim(1e-3, None)
            plt.yscale("symlog", linthresh=1e-3)

        # Create separate legends for iterative and gradient descent methods
        # First create dummy lines for the iterative methods legend
        iterative_legend_elements = [
            plt.Line2D(
                [0],
                [0],
                color="black",
                linestyle=style,
                linewidth=_df[_df["iterative_method"] == method].iloc[0]["linewidth"],
                label=NICE_NAMES[method],
            )
            for method, style in STYLES.items()
            if method != "None"
        ]

        # Create dummy patches for the gradient descent methods legend
        graddesc_legend_elements = [
            plt.Line2D(
                [0],
                [0],
                color=color,
                linestyle="-",
                linewidth=_df[_df["graddesc_method"] == method].iloc[0]["linewidth"],
                label=NICE_NAMES[method],
            )
            for method, color in COLORS.items()
        ]

        # Add both legends
        fig.legend(
            handles=iterative_legend_elements,
            title="Iterative Methods",
            bbox_to_anchor=(0.94, 0.7),
            loc="center left",
            borderaxespad=0,
        )
        fig.legend(
            handles=graddesc_legend_elements,
            title="Gradient Descent Methods",
            bbox_to_anchor=(0.94, 0.3),
            loc="center left",
            borderaxespad=0,
        )

        plt.ylabel(
            f"{PROP_NICE_NAMES[prop]}{'\n(instantaneous, relative to truncation)' if normalize_to_truncation else ''}",
            fontsize=FONTSIZE_LARGE,
            x=-0.02,
        )
        plt.xlabel("Time", fontsize=FONTSIZE_LARGE, y=-0.02)

        # # Calculate and plot line of best fit
        # x = method_data[xprop]
        # y = method_data[prop]

        # # Fit linear regression
        # slope, intercept = np.polyfit(x, y, 1)
        # line = slope * x + intercept

        # # Calculate R^2
        # correlation_matrix = np.corrcoef(x, y)
        # correlation_xy = correlation_matrix[0,1]
        # r_squared = correlation_xy**2

        # plt.plot(x, line, '--', label=f'{method} (R² = {r_squared:.3f})')

        plt.savefig(
            plots_folder / f"instantaneous_errors_over_time_{prop}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            plots_folder / f"instantaneous_errors_over_time_{prop}.pdf", bbox_inches="tight"
        )
        plt.show()


# %%


def plot_stacked_methods(
    df,
    palette,
    normalize_to_truncation=True,
    better_lower=True,
    padding=0.2,
    show_do_nothing=False,
    ymax=1.5,
    log_scale=False,
    hatch="\\\\\\\\\\",
    show_errorbars=True,
):
    omit_methods = [
        "bell_keep_classical__graddesc_global_reconstruction_split_non_interfering",
        "bell_keep_classical__rho_LM_MR_trace_norm",
        "bell_keep_classical__rho_half_LM_MR_trace_norm",
        "bell_keep_classical_old__rho_LM_MR_trace_norm_identical_blocks_old",
        "bell_discard_classical_old__rho_LM_MR_trace_norm_discard_classical_identical_blocks_old",
        "bell_keep_classical__rho_LM_MR_trace_norm_identical_blocks_old",
        "bell_discard_classical__rho_LM_MR_trace_norm_discard_classical_identical_blocks_old",
        "bell_discard_classical_old",
        "bell_keep_classical_old",
    ]

    if not show_do_nothing:
        omit_methods.append("do_nothing")

    # Define the two properties we'll stack
    prop_bottom = "global_reconstruction_error_trace_distance"
    prop_top = "interference_error_trace_distance"

    _df = copy.deepcopy(df)
    _df = _df.reset_index(drop=True)

    _df = _df[[x not in omit_methods for x in _df["method_name"]]]

    graddesc_methods = _df["graddesc_method"].unique()
    color_map = {x: get_color(x) for x in graddesc_methods}

    scatter_prop = "second-dominant TM eigenvalue magnitude"

    min_scatter_prop = min(_df[scatter_prop])
    max_scatter_prop = max(_df[scatter_prop])

    if normalize_to_truncation:
        # Create normalized copies of both properties
        normalized_props_bottom = copy.deepcopy(_df[prop_bottom])
        normalized_props_top = copy.deepcopy(_df[prop_top])

        for i, row in _df.iterrows():
            timestamp = row["timestamp"]
            df_now = _df[_df["timestamp"] == timestamp]
            truncation_prop = df_now[df_now["method_name"] == "truncation"][prop_bottom].item()
            # Use the same scaling factor for both properties
            normalized_props_bottom.loc[i] = row[prop_bottom] / truncation_prop
            normalized_props_top.loc[i] = row[prop_top] / truncation_prop

        _df[prop_bottom + "_normalized"] = normalized_props_bottom
        _df[prop_top + "_normalized"] = normalized_props_top

    iterative_methods = _df.iterative_method.unique()
    n_scatter_points = len(_df[_df["method_name"] == "truncation"])
    width_ratios = [
        (len(_df[_df.iterative_method == iterative_method]) + 2 * n_scatter_points * padding)
        for iterative_method in iterative_methods
    ]
    width_ratios[0] = width_ratios[0] + 2 * n_scatter_points * padding
    fig, ax = plt.subplots(
        1,
        len(iterative_methods),
        figsize=(15, 5),
        sharey=True,
        width_ratios=width_ratios,
        gridspec_kw={"wspace": 0},
    )

    for i, iterative_method in enumerate(iterative_methods):
        iterative_method_df = _df[_df.iterative_method == iterative_method].sort_values(
            by="graddesc_method"
        )
        if not show_do_nothing:
            iterative_method_df = iterative_method_df[
                iterative_method_df["method_name"] != "do_nothing"
            ]
        order = [
            k for k in graddesc_methods if k in iterative_method_df["graddesc_method"].unique()
        ]

        # Calculate means and standard errors for each graddesc_method
        stats = {}
        for method in order:
            graddesc_method_df = iterative_method_df[
                iterative_method_df["graddesc_method"] == method
            ]

            bottom_vals = (
                graddesc_method_df[prop_bottom]
                if not normalize_to_truncation
                else graddesc_method_df[prop_bottom + "_normalized"]
            )
            top_vals = (
                graddesc_method_df[prop_top]
                if not normalize_to_truncation
                else graddesc_method_df[prop_top + "_normalized"]
            )

            # Bootstrap confidence intervals
            def bootstrap_ci(data, n_bootstrap=1000):
                bootstrap_means = np.zeros(n_bootstrap)
                for j in range(n_bootstrap):
                    bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                    bootstrap_means[j] = np.mean(bootstrap_sample)
                return np.percentile(bootstrap_means, [2.5, 97.5])

            bottom_ci = bootstrap_ci(bottom_vals)
            top_ci = bootstrap_ci(top_vals)

            stats[method] = {
                "bottom_mean": np.mean(bottom_vals),
                "bottom_ci": bottom_ci,
                "top_mean": np.mean(top_vals),
                "top_ci": top_ci,
            }

        # Plot bottom bars
        x_pos = np.arange(len(order))
        bottom_means = [stats[m]["bottom_mean"] for m in order]
        bottom_cis = np.array([stats[m]["bottom_ci"] for m in order])
        bottom_yerr = np.abs(bottom_cis - np.array(bottom_means)[:, np.newaxis]).T

        ax[i].bar(
            x_pos,
            bottom_means,
            yerr=bottom_yerr if show_errorbars else None,
            width=1.0,
            color=[color_map[m] for m in order],
            alpha=1.0,
            edgecolor="white",
            capsize=0,
            error_kw={"alpha": 0.7, "ecolor": "black", "linewidth": 4.0},
        )

        # Plot top bars
        top_means = [stats[m]["top_mean"] for m in order]
        top_cis = np.array([stats[m]["top_ci"] for m in order])
        top_yerr = np.abs(top_cis - np.array(top_means)[:, np.newaxis]).T

        ax[i].bar(
            x_pos,
            top_means,
            yerr=top_yerr if show_errorbars else None,
            bottom=bottom_means,
            width=1.0,
            color=[color_map[m] for m in order],
            #  alpha=0.4,
            capsize=0,
            error_kw={"alpha": 0.7, "ecolor": "black", "linewidth": 4.0},
            hatch=hatch,
        )

        # Add scatter points for individual datapoints
        for j, method in enumerate(order):
            graddesc_method_df = iterative_method_df[
                iterative_method_df["graddesc_method"] == method
            ]
            bottom_vals = (
                graddesc_method_df[prop_bottom]
                if not normalize_to_truncation
                else graddesc_method_df[prop_bottom + "_normalized"]
            )
            top_vals = (
                graddesc_method_df[prop_top]
                if not normalize_to_truncation
                else graddesc_method_df[prop_top + "_normalized"]
            )
            total_vals = np.array(bottom_vals) + np.array(top_vals)
            assert len(total_vals) == len(top_vals), f"{len(total_vals)} != {len(top_vals)}"
            assert len(total_vals) == len(bottom_vals), f"{len(total_vals)} != {len(bottom_vals)}"

            # Sort the total_vals by the second-dominant TM eigenvalue magnitude
            sorted_indices = np.argsort(graddesc_method_df[scatter_prop])
            total_vals = total_vals[sorted_indices]
            scatter_prop_vals = np.array(graddesc_method_df[scatter_prop])[sorted_indices]

            ax[i].scatter(
                np.linspace(j - 0.4, j + 0.4, len(total_vals)),
                total_vals,
                c=scatter_prop_vals,
                cmap=sns.color_palette("blend:#CCC,#000", as_cmap=True),
                edgecolor="white",
                linewidth=0.8,
                s=15,
                alpha=1.0,
                vmin=min_scatter_prop,
                vmax=max_scatter_prop,
            )

        # Extend x-axis range on both sides to add spacing between subplots
        xlim = ax[i].get_xlim()
        ax[i].set_xlim(xlim[0] - padding, xlim[1] + padding)
        if i == 0:
            ax[i].set_xlim(xlim[0] - padding, xlim[1] + 3 * padding)

        ax[i].set_xlabel("")
        ax[i].set_xticks([])
        ax[i].set_title(get_nice_name(iterative_method), fontsize=FONTSIZE_SMALL, y=-0.09)
        if i == 0:
            ax[i].set_title(
                get_nice_name(iterative_method) + "     ", fontsize=FONTSIZE_SMALL, y=-0.09
            )

        if normalize_to_truncation:
            ax[i].axhline(y=1, color="black", linestyle="--", linewidth=0.8)

        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["bottom"].set_visible(False)
        ax[i].spines["left"].set_visible(False)

        if log_scale:
            ax[i].set_yscale("log")
            ax[i].yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            ax[i].grid(True, which="minor", alpha=0.25)
            ax[i].grid(True, which="major", alpha=1)

            ylim = ax[i].get_ylim()
            if ylim[0] < 1e-10:
                ax[i].set_ylim(1e-3, ylim[1])
            if "ising" in results_path:
                ax[i].set_ylim(1e-1, ylim[1] * 3)
        else:
            ax[i].set_ylim(0, ymax)

    ax[0].set_ylabel(
        "Global errors" + ("\n(relative to truncation)" * normalize_to_truncation),
        fontsize=FONTSIZE_LARGE,
    )
    plt.suptitle("Iterative method", y=0.03, fontsize=FONTSIZE_LARGE)

    # Create legend elements for gradient descent methods
    graddesc_legend_elements = [
        # Line2D([0], [0], marker='o', color='w',
        #        label=NICE_NAMES[graddesc_method],
        #        markerfacecolor=color,
        #        markersize=10)
        plt.Rectangle((0, 0), 1, 1, facecolor=color, label=get_nice_name(graddesc_method))
        for graddesc_method, color in color_map.items()
    ]

    # Create legend elements for error components
    error_legend_elements = [
        Patch(facecolor="gray", alpha=1.0, label="Interference error", hatch=hatch),
        Patch(facecolor="gray", alpha=1.0, label="Reconstruction error"),
    ]

    # Create legend elements for scatter plot
    scatter_legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"{min_scatter_prop:.2f}",
            markerfacecolor="#CCC",
            markersize=5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"{(min_scatter_prop + max_scatter_prop) / 2:.2f}",
            markerfacecolor=sns.color_palette("blend:#CCC,#000", as_cmap=True)(0.5),
            markersize=5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"{max_scatter_prop:.2f}",
            markerfacecolor="#000",
            markersize=5,
        ),
    ]

    # Add gradient descent methods legend first
    legend_graddesc = plt.legend(
        handles=graddesc_legend_elements,
        title="Gradient descent method",
        bbox_to_anchor=(-0.78, -0.2),
        loc="upper center",
        frameon=True,
        ncol=len(graddesc_legend_elements) // 2 + len(graddesc_legend_elements) % 2,
        fontsize=FONTSIZE_SMALL,
        title_fontsize=FONTSIZE_LARGE,
    )

    # Add error components legend second
    legend_err = plt.legend(
        handles=error_legend_elements,
        title="Error components (trace distance)",
        bbox_to_anchor=(-2.42, 1.0),
        loc="upper left",
        frameon=True,
        ncol=2,
        fontsize=FONTSIZE_SMALL,
        title_fontsize=FONTSIZE_SMALL,
    )

    # Add scatter plot legend third
    legend_scatter = plt.legend(
        handles=scatter_legend_elements,
        title="Second-dominant transfer-matrix eigenvalue",
        bbox_to_anchor=(1.0, 1.0),
        loc="upper right",
        frameon=True,
        ncol=3,
        fontsize=FONTSIZE_SMALL,
        title_fontsize=FONTSIZE_SMALL,
    )

    # Add all legends to the plot
    fig.add_artist(legend_graddesc)
    fig.add_artist(legend_err)

    if better_lower:
        ytrunc = 0.43
        plt.figtext(
            0.91,
            ytrunc + 0.105,
            "worse →",
            rotation=90,
            va="center",
            fontsize=FONTSIZE_SMALL,
            color="grey",
        )
        plt.figtext(
            0.91,
            ytrunc - 0.105,
            "← better",
            rotation=90,
            va="center",
            fontsize=FONTSIZE_SMALL,
            color="grey",
        )

    return fig, ax


if __name__ == "__main__":
    # Example usage:
    # sns.set_style('whitegrid')
    plot_stacked_methods(
        results, list(sns.color_palette("Paired")), log_scale=True, show_errorbars=show_errorbars
    )
    plt.savefig(plots_folder / "stacked_global_errors.png", dpi=300, bbox_inches="tight")
    plt.savefig(plots_folder / "stacked_global_errors.pdf", bbox_inches="tight")
    plt.show()


# %%


def plot_methods(
    df,
    prop,
    palette,
    normalize_to_truncation=True,
    better_lower=True,
    padding=0.2,
    show_do_nothing=False,
    ymax=1.5,
    log_scale=False,
    show_errorbars=True,
):
    omit_methods = [
        "bell_keep_classical__graddesc_global_reconstruction_split_non_interfering",
        "bell_keep_classical__rho_LM_MR_trace_norm",
        "bell_keep_classical__rho_half_LM_MR_trace_norm",
        "bell_keep_classical_old__rho_LM_MR_trace_norm_identical_blocks_old",
        "bell_discard_classical_old__rho_LM_MR_trace_norm_discard_classical_identical_blocks_old",
        "bell_keep_classical__rho_LM_MR_trace_norm_identical_blocks_old",
        "bell_discard_classical__rho_LM_MR_trace_norm_discard_classical_identical_blocks_old",
        "bell_discard_classical_old",
        "bell_keep_classical_old",
    ]
    if not show_do_nothing:
        omit_methods.append("do_nothing")

    if any(df[df["method_name"] == "truncation"][prop]) == 0.0 or "norm" in prop:
        normalize_to_truncation = False
    if "norm" in prop:
        better_lower = False
        log_scale = False
        ymax = 0.6

    _df = copy.deepcopy(df)
    _df = _df.reset_index(drop=True)
    _df = _df[[x not in omit_methods for x in _df["method_name"]]]

    graddesc_methods = _df["graddesc_method"].unique()
    color_map = {x: get_color(x) for x in graddesc_methods}

    scatter_prop = "second-dominant TM eigenvalue magnitude"
    min_scatter_prop = min(_df[scatter_prop])
    max_scatter_prop = max(_df[scatter_prop])

    normalized_props = copy.deepcopy(_df[prop])
    if normalize_to_truncation:
        for i, row in _df.iterrows():
            timestamp = row["timestamp"]
            df_now = _df[_df["timestamp"] == timestamp]
            truncation_prop = df_now[df_now["method_name"] == "truncation"][prop].item()
            normalized_props.loc[i] = row[prop] / truncation_prop

    iterative_methods = _df.iterative_method.unique()
    n_scatter_points = len(_df[_df["method_name"] == "truncation"])
    width_ratios = [
        (len(_df[_df.iterative_method == iterative_method]) + 2 * n_scatter_points * padding)
        for iterative_method in iterative_methods
    ]
    width_ratios[0] = width_ratios[0] + 2 * n_scatter_points * padding
    fig, ax = plt.subplots(
        1,
        len(iterative_methods),
        figsize=(15, 5),
        sharey=True,
        width_ratios=width_ratios,
        gridspec_kw={"wspace": 0},
    )

    for i, iterative_method in enumerate(iterative_methods):
        iterative_method_df = _df[_df.iterative_method == iterative_method].sort_values(
            by="graddesc_method"
        )
        if not show_do_nothing:
            iterative_method_df = iterative_method_df[
                iterative_method_df["method_name"] != "do_nothing"
            ]
        order = [
            k for k in graddesc_methods if k in iterative_method_df["graddesc_method"].unique()
        ]

        # Calculate means and standard errors for each graddesc_method
        stats = {}
        for method in order:
            graddesc_method_df = iterative_method_df[
                iterative_method_df["graddesc_method"] == method
            ]
            vals = (
                graddesc_method_df[prop]
                if not normalize_to_truncation
                else normalized_props[graddesc_method_df.index]
            )

            # Bootstrap confidence intervals
            def bootstrap_ci(data, n_bootstrap=1000):
                bootstrap_means = np.zeros(n_bootstrap)
                for j in range(n_bootstrap):
                    bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                    bootstrap_means[j] = np.mean(bootstrap_sample)
                return np.percentile(bootstrap_means, [2.5, 97.5])

            ci = bootstrap_ci(vals)
            stats[method] = {"mean": np.mean(vals), "ci": ci}

        # Plot bars
        x_pos = np.arange(len(order))
        means = [stats[m]["mean"] for m in order]
        cis = np.array([stats[m]["ci"] for m in order])
        yerr = np.abs(cis - np.array(means)[:, np.newaxis]).T

        ax[i].bar(
            x_pos,
            means,
            yerr=yerr if show_errorbars else None,
            width=1.0,
            color=[color_map[m] for m in order],
            alpha=1.0,
            edgecolor="white",
            capsize=0,
            error_kw={"alpha": 0.7, "ecolor": "black", "linewidth": 4.0},
        )

        # Add scatter points for individual datapoints
        for j, method in enumerate(order):
            graddesc_method_df = iterative_method_df[
                iterative_method_df["graddesc_method"] == method
            ]
            vals = (
                graddesc_method_df[prop]
                if not normalize_to_truncation
                else normalized_props[graddesc_method_df.index]
            )

            # Sort by scatter_prop
            sorted_indices = np.argsort(np.array(graddesc_method_df[scatter_prop]))
            vals = np.array(vals)[sorted_indices]
            scatter_prop_vals = np.array(graddesc_method_df[scatter_prop])[sorted_indices]

            ax[i].scatter(
                np.linspace(j - 0.4, j + 0.4, len(vals)),
                vals,
                c=scatter_prop_vals,
                cmap=sns.color_palette("blend:#CCC,#000", as_cmap=True),
                edgecolor="white",
                linewidth=0.8,
                s=15,
                alpha=1.0,
                vmin=min_scatter_prop,
                vmax=max_scatter_prop,
            )

        # Extend x-axis range on both sides to add spacing between subplots
        xlim = ax[i].get_xlim()
        ax[i].set_xlim(xlim[0] - padding, xlim[1] + padding)
        if i == 0:
            ax[i].set_xlim(xlim[0] - padding, xlim[1] + 3 * padding)

        ax[i].set_xlabel("")
        ax[i].set_xticks([])
        ax[i].set_title(
            get_nice_name(iterative_method).capitalize(), fontsize=FONTSIZE_SMALL, y=-0.09
        )
        if i == 0:
            ax[i].set_title(
                get_nice_name(iterative_method) + "     ", fontsize=FONTSIZE_SMALL, y=-0.09
            )

        if normalize_to_truncation:
            ax[i].axhline(y=1, color="black", linestyle="--", linewidth=0.8)

        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["bottom"].set_visible(False)
        # ax[i].spines['left'].set_visible(False)

        if log_scale:
            ax[i].set_yscale("log")
            ax[i].yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            ax[i].grid(True, which="minor", alpha=0.25)
            ax[i].grid(True, which="major", alpha=1)

            ylim = ax[i].get_ylim()
            # if ylim[0] < 1e-2:
            ax[i].set_ylim(2e-2, ylim[1])
        else:
            ax[i].set_ylim(0, ymax)

    ax[0].set_ylabel(
        PROP_NICE_NAMES[prop] + ("\n(relative to truncation)" * normalize_to_truncation),
        fontsize=FONTSIZE_LARGE,
    )
    plt.suptitle("Iterative method", y=0.03, fontsize=FONTSIZE_LARGE)

    # Create legend elements for gradient descent methods
    graddesc_legend_elements = [
        # Line2D([0], [0], marker='o', color='w',
        #        label=NICE_NAMES[graddesc_method],
        #        markerfacecolor=color,
        #        markersize=10)
        plt.Rectangle((0, 0), 1, 1, facecolor=color, label=get_nice_name(graddesc_method))
        for graddesc_method, color in color_map.items()
    ]

    # Create legend elements for scatter plot
    scatter_legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"{min_scatter_prop:.2f}",
            markerfacecolor="#CCC",
            markersize=5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"{(min_scatter_prop + max_scatter_prop) / 2:.2f}",
            markerfacecolor=sns.color_palette("blend:#CCC,#000", as_cmap=True)(0.5),
            markersize=5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"{max_scatter_prop:.2f}",
            markerfacecolor="#000",
            markersize=5,
        ),
    ]

    # Add gradient descent methods legend
    legend_graddesc = plt.legend(
        handles=graddesc_legend_elements,
        title="Gradient descent method",
        #   bbox_to_anchor=(-1.0, -0.2),
        bbox_to_anchor=(-0.78, -0.2),
        loc="upper center",
        frameon=True,
        ncol=len(graddesc_legend_elements) // 2 + len(graddesc_legend_elements) % 2,
        fontsize=FONTSIZE_SMALL,
        title_fontsize=FONTSIZE_LARGE,
    )

    # Add scatter plot legend
    legend_scatter = plt.legend(
        handles=scatter_legend_elements,
        title="Second-dominant transfer-matrix eigenvalue",
        bbox_to_anchor=(1.0, 1.0),
        loc="upper right",
        frameon=True,
        ncol=3,
        fontsize=FONTSIZE_SMALL,
        title_fontsize=FONTSIZE_SMALL,
    )

    # Add all legends to the plot
    fig.add_artist(legend_graddesc)

    if better_lower:
        ytrunc = 0.43
        plt.figtext(
            0.91,
            ytrunc + 0.105,
            "worse →",
            rotation=90,
            va="center",
            fontsize=FONTSIZE_SMALL,
            color="grey",
        )
        plt.figtext(
            0.91,
            ytrunc - 0.105,
            "← better",
            rotation=90,
            va="center",
            fontsize=FONTSIZE_SMALL,
            color="grey",
        )

    return fig, ax


# sns.set_style('whitegrid')

if __name__ == "__main__":
    for prop in [
        # 'global_reconstruction_error_trace_distance',
        # 'interference_error_trace_distance',
        # 'reconstruction_plus_interference_error_trace_distance',
        "costFun_LM_MR_trace_distance",
        # # 'costFun_split_trace_distance',
        # # 'one_minus_overlap_pure',
        # # 'costFun_LM_MR_rho_half_frobenius',
        "norms_geometric_mean",
    ]:
        plot_methods(
            results, prop, palette, log_scale=True, show_errorbars=show_errorbars
        )  # , ymax=0.6, log_scale=False)
        plt.savefig(plots_folder / f"{prop}_barscatter.png", dpi=300, bbox_inches="tight")
        plt.savefig(plots_folder / f"{prop}_barscatter.pdf", bbox_inches="tight")
        plt.show()
# %%
# Create a scatterplot showing which methods are both better than truncation in global reconstruction error and better than truncation in interference error
if __name__ == "__main__":
    plt.figure(figsize=(5, 5), dpi=180)
    relative_to_truncation = True
    yprop = "costFun_LM_MR_trace_distance"
    xprop = "global_reconstruction_error_trace_distance"
    # xprop = 'reconstruction_plus_interference_error_trace_distance'
    df = copy.deepcopy(results)
    df = df[df[yprop] > 1e-10]
    df = df[df[xprop] > 1e-10]
    # Plot
    normalized_yprops = copy.deepcopy(df[yprop])
    normalized_xprops = copy.deepcopy(df[xprop])
    if relative_to_truncation:
        for i, row in df.iterrows():
            timestamp = row["timestamp"]
            df_now = df[df["timestamp"] == timestamp]
            truncation_yprop = df_now[df_now["method_name"] == "truncation"][yprop].item()
            normalized_yprops.loc[i] = row[yprop] / truncation_yprop
            truncation_xprop = df_now[df_now["method_name"] == "truncation"][xprop].item()
            normalized_xprops.loc[i] = row[xprop] / truncation_xprop
    else:
        normalized_yprops = copy.deepcopy(df[yprop])
        normalized_xprops = copy.deepcopy(df[xprop])

    plt.scatter(
        normalized_xprops,
        normalized_yprops,
        s=5,
        alpha=0.5,
        c=[get_color(x) for x in df["graddesc_method"]],
    )
    plt.ylabel(PROP_NICE_NAMES[yprop] + "\n(relative to truncation)" * relative_to_truncation)
    plt.xlabel(PROP_NICE_NAMES[xprop] + "\n(relative to truncation)" * relative_to_truncation)
    plt.yscale("symlog", linthresh=1e-8)
    plt.xscale("symlog", linthresh=1e-3)

    graddesc_methods = df["graddesc_method"].unique()
    color_map = {x: get_color(x) for x in graddesc_methods}
    # Create legend elements for gradient descent methods
    graddesc_legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=get_nice_name(graddesc_method),
            markerfacecolor=color,
            alpha=0.75,
            markersize=10,
        )
        for graddesc_method, color in color_map.items()
    ]

    # Add gradient descent methods legend first
    legend_graddesc = plt.legend(
        handles=graddesc_legend_elements,
        title="Gradient descent method",
        bbox_to_anchor=(1.0, 0.5),
        loc="center left",
        frameon=True,
        #   ncol=len(graddesc_legend_elements)//2 + len(graddesc_legend_elements)%2,
        fontsize=FONTSIZE_SMALL,
        title_fontsize=FONTSIZE_LARGE,
    )
    plt.show()


# %%
################################################################
#########################  REAL DATA  ##########################
################################################################
if __name__ == "__main__":
    normalize_to_truncation = False
    F = sns.catplot(
        results,
        kind="bar",
        x="iterative_method",
        y="costFun_LM_MR_trace_distance",
        hue="graddesc_method",
        palette=palette,
        legend_out=False,
        aspect=2,
        height=5,
    )
    g = F.ax
    # g.despine(left=True)
    plt.xticks(rotation=30, ha="right", rotation_mode="anchor")
    # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    labels = [item.get_text().replace("_", " ") for item in g.get_xticklabels()]
    g.set_xticklabels(labels)
    ylabel = g.get_ylabel().replace("_", " ")
    if normalize_to_truncation:
        ylabel += "  (normalized to truncation)"
    g.set_ylabel(ylabel)
    xlabel = g.get_xlabel().replace("_", " ")
    g.set_xlabel(xlabel)
    plt.show()

    ################################################################
    #########################  REAL DATA  ##########################
    ################################################################
    # %%
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = [10, 8]
    for prop in [
        # # "trace_distance_L",
        # # "trace_distance_Lpl",
        # # "trace_distance_LR",
        # # "trace_distance_LMR",
        # # "norm_error_pure",
        # # "norm_error_mixed",
        # # "overlap_LR_branches",
        # # "overlap_branches",
        # "one_minus_overlap_pure",
        # "one_minus_overlap_mixed",
        # # "svals_entropy_l_sum",
        # # "svals_entropy_m_sum",
        # # "svals_entropy_r_sum",
        # # "svals_entropy_l_sum_on_orig",
        # # "svals_entropy_m_sum_on_orig",
        # # "svals_entropy_r_sum_on_orig",
        # # "records_quality_left",
        # # "trace_distance_M",
        # # "trace_distance_R",
        # # "trace_distance_LM",
        # # "trace_distance_Rpr",
        # # "trace_distance_MR",
        "costFun_LM_MR_trace_distance",
        "costFun_split_trace_distance",
        "global_reconstruction_error_trace_distance",
        "interference_error_trace_distance",
        "reconstruction_plus_interference_error_trace_distance",
        # "costFun_LM_MR_rho_half_frobenius",
        "norms_geometric_mean",
        # "norms_prod",
        # "overlap_branches",
        # "overlap_LR_branches",
        # "〈σx〉_error",
        # # "〈σx σx〉_error",
        # # "norm_orig",
        # # "norm_branch_0",
        # # "norm_branch_1",
        # # "norm_zeroed_recon",
        # # "〈σx〉_zeroed_error",
        # # "〈σx〉_branches_combined_error",
        # # "rho_branch_0_trace_norm_error",
        # # "rho_branch_1_trace_norm_error",
        # # "rho_zeroed_trace_norm_error",
        # # "rho_branches_combined_trace_norm_error",
        # # "expected_bond_dim_reduction_factor"
        # # "rho_zeroed_vs_branches_combined_trace_norm_error",
        # # "tr(rho)_zeroed_recon",
        # # "〈σx σx〉_zeroed_error",
        # # "〈σx σx〉_branches_combined_error",
        # # "frobenius_error",
        # # "trace_norm_error",
        # # "rho_trace_norm_error",
        # # "rho_trace_norm_error_site_A",
        # # "rho_trace_norm_error_site_B",
        # # # "offblock_error"
        # # "tr(rho_orig.H @ rho_zeroed_recon)",
    ]:
        plot_proportion_beating_truncation = False

        prop_safe = (
            prop.replace("〈", "")
            .replace("〉", "")
            .replace("(", "_")
            .replace(")", "")
            .replace("σ", "sigma_")
        )
        name = prop_safe  # + "_vs_t"
        print(name)
        forms = results["form_str"].unique()
        for form in forms:
            if form != "null":
                df = copy.deepcopy(results)
                df = results[results["form_str"] == form]
                # df = df[~df["kind"].str.contains('g0=1.0')]
                normalized_props = copy.deepcopy(df[prop])
                ns_total = defaultdict(int)
                ns_beating_truncation = defaultdict(int)
                ns_beating_truncation_by_factor_of_2 = defaultdict(int)
                if "norm" not in prop and plot_proportion_beating_truncation:
                    for i, row in df.iterrows():
                        timestamp = row["timestamp"]
                        df_now = df[df["timestamp"] == timestamp]
                        truncation_prop = df_now[df_now["method_name"] == "truncation"][prop].item()
                        normalized_props.loc[i] = row[prop] / truncation_prop
                        ns_total[row["method_name"]] += 1
                        ns_beating_truncation[row["method_name"]] += int(
                            row[prop] < truncation_prop
                        )
                        ns_beating_truncation_by_factor_of_2[row["method_name"]] += int(
                            row[prop] < 0.5 * truncation_prop
                        )

                    proportion_beating_truncation = {
                        key: ns_beating_truncation[key] / ns_total[key] for key in ns_total
                    }
                    proportion_beating_truncation_by_factor_of_2 = {
                        key: ns_beating_truncation_by_factor_of_2[key] / ns_total[key]
                        for key in ns_total
                    }
                    # Plot the proportion beating truncation
                    plt.figure(dpi=180)
                    g = sns.barplot(
                        x=list(proportion_beating_truncation.keys()),
                        y=list(proportion_beating_truncation.values()),
                        palette=palette,
                    )
                    plt.ylabel(name)
                    plt.xticks(rotation=30, ha="right", rotation_mode="anchor")
                    labels = [item.get_text().replace("_", " ") for item in g.get_xticklabels()]
                    g.set_xticklabels(labels)
                    ylabel = g.get_ylabel().replace("_", " ")
                    g.set_ylabel(f"Proportion beating truncation on {ylabel}")
                    xlabel = g.get_xlabel().replace("_", " ")
                    g.set_xlabel(xlabel)
                    plt.savefig(
                        plots_folder / (name + "_proportion-beating-truncation" + ".png"),
                        bbox_inches="tight",
                    )
                    plt.show()

                    # Plot the proportion beating truncation by a factor of 2
                    plt.figure(dpi=180)
                    g = sns.barplot(
                        x=list(proportion_beating_truncation_by_factor_of_2.keys()),
                        y=list(proportion_beating_truncation_by_factor_of_2.values()),
                        palette=palette,
                    )
                    plt.ylabel(name)
                    plt.xticks(rotation=30, ha="right", rotation_mode="anchor")
                    labels = [item.get_text().replace("_", " ") for item in g.get_xticklabels()]
                    g.set_xticklabels(labels)
                    ylabel = g.get_ylabel().replace("_", " ")
                    g.set_ylabel(f"Proportion beating truncation by a factor of 2 on {ylabel}")
                    xlabel = g.get_xlabel().replace("_", " ")
                    g.set_xlabel(xlabel)
                    plt.savefig(
                        plots_folder
                        / (name + "_proportion-beating-truncation-factor-of-2" + ".png"),
                        bbox_inches="tight",
                    )
                    plt.show()

                # Plot the normalized properties
                plt.figure(dpi=180)
                # df = df[df['from_frias_perez'] == "false"]
                # g = sns.lineplot(
                #     df,
                #     y=df[prop],
                #     x = np.round(df['t'], decimals=2),
                #     hue="method_name",
                #     style="method_name",
                #     alpha=0.3,
                #     legend=False
                # )
                g = sns.barplot(
                    df,
                    y=normalized_props,
                    # x = np.round(df['t'], decimals=2),
                    # x = np.round(df['second-dominant TM eigenvalue magnitude'], decimals=2),
                    x="method_name",
                    palette=palette,
                    # x = df['average bond dimension'],
                    # hue="method_name",
                    # col="correlation length",
                    # height=7,
                    # aspect=1.5,
                    # kind="bar",
                    # sharex=False,
                    # size=1.0/(df["compression_ratio"])-1.0,
                    # alpha=1,
                )
                # g3 = sns.pointplot(data=df, x="method_name", y=normalized_props, hue="kind", dodge=0.55, join=False, #markers=["o", "x", "s", "D"],
                #     palette = sns.color_palette("Spectral", n_colors=len(df['kind'].unique())),#sns.color_palette("blend:#CCC,#000", as_cmap=True),
                #     )
                g2 = sns.stripplot(
                    df,
                    y=normalized_props,
                    # x = np.round(df['t'], decimals=2),
                    # x = np.round(df['second-dominant TM eigenvalue magnitude'], decimals=2),
                    x="method_name",
                    # x = df['average bond dimension'],
                    hue="second-dominant TM eigenvalue magnitude",  # kind
                    palette=sns.color_palette(
                        "blend:#CCC,#000", as_cmap=True
                    ),  # sns.color_palette("Spectral", n_colors=len(df['kind'].unique())),#
                    edgecolor="white",
                    linewidth=0.8,
                    dodge=True,
                    jitter=False,
                    # col="correlation length",
                    # height=7,
                    # aspect=1.5,
                    # kind="bar",
                    # sharex=False,
                    # size=1.0/(df["compression_ratio"])-1.0,
                    # alpha=0.5,
                    size=3,
                )
                # g3 = sns.pointplot(data=df, x="method_name", y=prop, hue="second-dominant TM eigenvalue magnitude", dodge=True, scale=0.2,
                #     palette= sns.color_palette("blend:#CCC,#000"),
                #     )
                plt.ylabel(name)
                # plt.title(f'{name}  -  form = {form}')
                if "svals" not in prop:
                    # g.axes[0][0].set_yscale("symlog", linthresh=1e-8)
                    g.set_yscale("symlog", linthresh=1e-3)
                # plt.ylim(None, 2)
                plt.xticks(rotation=30, ha="right", rotation_mode="anchor")
                # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
                labels = [item.get_text().replace("_", " ") for item in g.get_xticklabels()]
                g.set_xticklabels(labels)
                ylabel = g.get_ylabel().replace("_", " ")
                if "norm" not in prop:
                    ylabel += "  (normalized to truncation)"
                g.set_ylabel(ylabel)
                xlabel = g.get_xlabel().replace("_", " ")
                g.set_xlabel(xlabel)
                g.legend(loc="lower left", title="|2nd-dominant TM eigenvalue|")
                plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
                plt.show()

    # %%
    ################################################################
    # NORMS OF EACH BRANCH
    ################################################################
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = [10, 8]
    plt.figure(dpi=200)
    i = 0
    for property in [
        "norm_branch_0",
        "norm_branch_1",
        "norm_branch_2",
        "norm_branch_3",
    ]:
        property_safe = (
            property.replace("〈", "")
            .replace("〉", "")
            .replace("(", "_")
            .replace(")", "")
            .replace("σ", "sigma_")
        )
        name = property_safe + "_vs_t"
        df = results
        # df = df[df['from_frias_perez'] == "false"]
        # g = sns.lineplot(
        #     df,
        #     y=df[property],
        #     x = np.round(df['t'], decimals=2),
        #     hue="method_name",
        #     style="method_name",
        #     alpha=0.3,
        #     legend = False
        # )
        g = sns.scatterplot(
            df,
            y=df[property],
            x=np.round(df["t"], decimals=2),
            hue="method_name",
            style="method_name",
            # size=1.0/(df["compression_ratio"])-1.0,
            alpha=1,
            legend=True if i == 0 else False,
        )
        i += 1
    plt.ylabel("norm")
    plt.title("Norms of each branch")
    # g.axes.set_yscale("logit")
    g.axes.set_yscale("symlog", linthresh=1e-10)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
    plt.show()

    # %%
    for property in ["expected_bond_dim_reduction_factor"]:
        property_safe = (
            property.replace("〈", "").replace("〉", "").replace("(", "_").replace(")", "")
        )
        name = property_safe + "_vs_t"
        print(name)
        forms = results["form_str"].unique()
        for form in forms:
            if form != "null":
                df = results[results["form_str"] == form]
                # df = df[df['from_frias_perez'] == "false"]
                g = sns.lineplot(
                    df,
                    y=property,
                    x="t",
                    hue="method_name",
                    style="from_frias_perez",
                    alpha=0.3,
                    legend=False,
                )
                g = sns.scatterplot(
                    df,
                    y=property,
                    x="t",
                    hue="method_name",
                    style="from_frias_perez",
                    size=1.0 / (df["compression_ratio"]) - 1.0,
                    alpha=1,
                )
                plt.title(f"form = {form}")
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
                plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
                plt.show()

    # %%
    name = "trace_norm_error_vs_dim_L_scatterplot"
    for form in forms:
        g = sns.scatterplot(
            results[results["form_str"] == form],
            y="trace_norm_error",
            x="dim_L",
            hue="method_name",
            alpha=0.75,
        )
        plt.title(f"form = {form}")
        g.axes.set_yscale("symlog", linthresh=1e-8)
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="upper left")
        plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
        plt.show()

    # %%

    ################################################################
    ##########################  TOY DATA  ##########################
    ################################################################
    # %%
    # sns.set_style("whitegrid")
    # sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})
    # sns.set_style("whitegrid")
    # data = results.loc[results['scramble_kind'].isin(['UU','UV'])]
    # data = data.loc[results['method_name'].isin(['bsvd_nograddesc','bsvd_fastest_nograddesc'])]
    g = sns.lineplot(
        results,
        y="walltime",
        x="dim_L",
        hue="method_name",
    )
    g = sns.scatterplot(results, y="walltime", x="dim_L", hue="method_name", alpha=1, legend=False)
    x = np.arange(40, 800, 10)
    # plt.plot(x, 5e-8*x**3, label="cubic")
    plt.xscale("log")
    plt.yscale("log")
    g.tick_params(which="both", bottom=True)

    plt.grid(True, which="both", ls="--", c="gray", alpha=0.2)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # plt.yticks(np.array([2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384])*0.001)
    # plt.xticks([32,64,128,256,512,1024])
    plt.savefig(plots_folder / ("extra_time_plot_log_log" + ".png"))
    plt.show()

    # %%
    name = "trace_norm_error_vs_noise_introduced_scatterplot"
    g = sns.catplot(
        results,
        y="trace_norm_error",
        x="noise_introduced",
        hue="method_name",
        jitter=0.3,
        dodge=True,
        alpha=0.7,
    )
    g.axes[0][0].set_yscale("symlog", linthresh=1e-9)
    plt.ylim(0, None)
    plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
    plt.show()

    # %%
    name = "trace_norm_error_vs_noise_introduced_scatterplot"
    g = sns.catplot(
        results,
        y="trace_norm_error",
        x="noise_introduced",
        col="kind",
        row="scramble_kind",
        hue="method_name",
        jitter=0.3,
        dodge=True,
        alpha=0.25,
    )
    g.axes[0][0].set_yscale("symlog", linthresh=1e-16)
    plt.ylim(0, None)
    plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
    plt.show()

    # %%
    name = "trace_norm_error_vs_noise_introduced_scatterplot_bell"
    g = sns.catplot(
        results,
        y="trace_norm_error",
        x="noise_introduced",
        col="bell_like",
        # row = "scramble_kind",
        hue="method_name",
        jitter=0.3,
        dodge=True,
        alpha=0.75,
    )
    g.axes[0][0].set_yscale("symlog", linthresh=1e-16)
    plt.ylim(0, None)
    plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
    plt.show()

    # %%
    name = "trace_norm_error_vs_noise_introduced_scatterplot_equal_sized_blocks"
    g = sns.catplot(
        results,
        y="trace_norm_error",
        x="noise_introduced",
        col="equal_sized_blocks",
        row="scramble_kind",
        hue="method_name",
        jitter=0.3,
        dodge=True,
        alpha=0.25,
    )
    g.axes[0][0].set_yscale("symlog", linthresh=1e-16)
    plt.ylim(0, None)
    plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
    plt.show()

    # %%
    name = "trace_norm_error_vs_noise_introduced_scatterplot_equal_square_blocks"
    g = sns.catplot(
        results,
        y="trace_norm_error",
        x="noise_introduced",
        col="square_blocks",
        row="scramble_kind",
        hue="method_name",
        jitter=0.3,
        dodge=True,
        alpha=0.25,
    )
    g.axes[0][0].set_yscale("symlog", linthresh=1e-16)
    plt.ylim(0, None)
    plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
    plt.show()

    # %%
    name = "offblock_error_vs_noise_introduced_scatterplot"
    g = sns.catplot(
        results,
        y="offblock_error",
        x="noise_introduced",
        col="kind",
        row="scramble_kind",
        hue="method_name",
        jitter=0.3,
        dodge=True,
        alpha=0.25,
    )
    g.axes[0][0].set_yscale("symlog", linthresh=1e-16)
    plt.ylim(0, None)
    plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
    plt.show()

    # %%
    name = "offblock_error_vs_noise_introduced_scatterplot_bell"
    g = sns.catplot(
        results,
        y="offblock_error",
        x="noise_introduced",
        col="bell_like",
        row="scramble_kind",
        hue="method_name",
        jitter=0.3,
        dodge=True,
        alpha=0.25,
    )
    g.axes[0][0].set_yscale("symlog", linthresh=1e-16)
    plt.ylim(0, None)
    plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
    plt.show()

    # %%

    name = "trace_norm_error_vs_compression"
    g = sns.catplot(
        results,
        y="trace_norm_error",
        x="compression_ratio",
        col="kind",
        row="scramble_kind",
        hue="method_name",
        jitter=0.3,
        dodge=True,
        alpha=0.25,
    )
    plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
    plt.show()

    # %%

    name = "offblock_error_vs_noise_introduced_barplot"
    g = sns.catplot(
        results.loc[results["scramble_kind"].isin(["UU", "UV"])],
        y="offblock_error",
        x="noise_introduced",
        col="kind",
        hue="method_name",
        kind="bar",
        dodge=True,
    )
    g.axes[0][0].set_yscale("symlog", linthresh=1e-15)
    plt.ylim(0, 1)
    plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
    plt.show()

    # %%
    for scale in ["log", "linear"]:
        name = f"offblock_error_vs_noise_introduced_lineplot_{scale}"
        g = sns.lineplot(
            results,  # .loc[results['scramble_kind'].isin(['UU','UV'])],
            y="offblock_error",
            x="noise_introduced",
            hue="method_name",
        )

        g = sns.scatterplot(
            results,  # .loc[results['scramble_kind'].isin(['UU','UV'])],
            y="offblock_error",
            x="noise_introduced",
            hue="method_name",
            alpha=0.1,
        )
        xs = results["noise_introduced"].unique().tolist()
        plt.plot(xs, xs, alpha=0.5, color="black", linewidth=1, linestyle="--")
        # g.xaxis.set_ticks(xs)
        if scale == "log":
            xs.append(1.0)
            g.set_yscale("symlog", linthresh=1e-15)
            g.set_xscale("symlog", linthresh=1e-15)
            plt.ylim(0, 1)
            plt.xlim(0, 1)
        plt.title(name)
        plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
        plt.show()

    # %%
    for dim_L in ["all"]:  # + results['dim_L'].unique().tolist():
        for scale in ["log", "linear"]:
            name = f"offblock_error_vs_noise_introduced_residual_factor_bond_dim_{dim_L}_{scale}"
            filtered = results  # .loc[results['scramble_kind'].isin(['UU','UV'])]
            if dim_L != "all":
                filtered = filtered.loc[results["dim_L"] == dim_L]
            y = results["offblock_error"] / results["noise_introduced"]
            x = "noise_introduced"
            g = sns.lineplot(
                filtered,
                y=y,
                x=x,
                hue="method_name",
            )

            g = sns.scatterplot(filtered, y=y, x=x, hue="method_name", alpha=0.3, legend=False)
            xs = results["noise_introduced"].unique().tolist()
            # g.xaxis.set_ticks(xs)
            if scale == "log":
                xs.append(1.0)
                g.set_yscale("log")
            g.set_xscale("log")
            plt.title(name)
            g.set_ylabel("offblock_error / noise_introduced")
            plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
            plt.show()

    # %%
    name = f"compression_vs_noise_introduced_lineplot_{scale}"
    g = sns.lineplot(
        results.loc[results["scramble_kind"].isin(["UU", "UV"])],
        y="compression_ratio",
        x="noise_introduced",
        hue="method_name",
    )
    plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
    plt.show()

    # %%
    for scale in ["log", "linear"]:
        name = f"offblock_error_vs_dim_L_lineplot_{scale}"
        g = sns.scatterplot(
            results,  # .loc[results['scramble_kind'].isin(['UU','UV'])],
            y="offblock_error",
            x="dim_L",
            hue="method_name",
        )
        if scale == "log":
            xs.append(1.0)
            g.set_yscale("symlog", linthresh=1e-15)
            # plt.ylim(0,1)
        plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
        plt.show()

    # %%
    ################################################################
    #########################  REAL DATA  ##########################
    ################################################################
    forms = results["form_str"].unique()

    for form in forms:
        for linear in [True, False]:
            name = f"offblock_error_vs_time_form-{form}_{'linear' if linear else 'logscale'}"
            g = sns.lmplot(
                results[results["form_str"] == form],
                palette="bright",
                y="offblock_error",
                x="t",
                hue="method_name",
                # markers = ['D', 's', 'o'],
                fit_reg=linear,
            )
            if not linear:
                g.axes[0][0].set_yscale("log")
            plt.title(name)
            plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
            plt.show()

        name = f"compression_vs_time-{form}"
        g = sns.lmplot(
            results[results["form_str"] == form],
            palette="bright",
            y="compression_ratio",
            x="t",
            hue="method_name",
            # markers = ['D', 's', 'o'],
            fit_reg=False,
        )
        plt.savefig(plots_folder / (name + ".png"), bbox_inches="tight")
        plt.title(name)
        plt.show()

    # %%

    # %%
