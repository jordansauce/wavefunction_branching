# %% # Plot the expectation values of the best run over time vs the truncation runs

import glob
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from natsort import natsorted

from evolution_analysis.error_analysis import OP_NAMES_LATEX
from evolution_analysis.pickle_analysis import DISTANCES, OP_NAMES_ANALYTIC
from wavefunction_branching.evolve_and_branch_finite import (
    BranchValues,
    extrapolate_analytic_results,
)

if __name__ == "__main__":
    DIRECTORY = Path(__file__).parent.parent

    truncation_runs = (
        "runs_truncation/2025-02-13-L80-chi100-at999999-n500-f_None/pickles/L80-chi100-at999999-n500-f_None_branch_values.pkl",
        "runs_truncation/2025-02-13-L80-chi200-at999999-n500-f_None/pickles/L80-chi200-at999999-n500-f_None_branch_values.pkl",
        "runs_truncation/2025-02-13-L80-chi400-at999999-n500-f_None/pickles/L80-chi400-at999999-n500-f_None_branch_values.pkl",
    )

    # best_run = "runs/2025-03-08-L80-chi100-at100-n400-f_vertical_svd_micro_bsvd__graddesc_global_reconstruction_split_non_interfering/pickles/L80-chi100-at100-n400-f_vertical_svd_micro_bsvd__graddesc_global_reconstruction_split_non_interfering_branch_values.pkl"
    # best_run_name = "Branching weighted average: bond dimension 100 \nvertical SVD + global reconstruction and 2-site non-interference"
    # save_suffix = ""

    best_run = "runs/2025-03-08-L80-chi100-at100-n400-f_bell_original_threshold_keep_classical__rho_LM_MR_trace_norm/pickles/L80-chi100-at100-n400-f_bell_original_threshold_keep_classical__rho_LM_MR_trace_norm_branch_values.pkl"
    best_run_name = "Branching weighted average: bond dimension 100 \nBell original threshold + Local trace distance"
    save_suffix = "_frias_perez_threshold"

    branch_values = pickle.load(open(DIRECTORY / best_run, "rb"))

    truncation_branch_values = []
    for truncation_run in truncation_runs:
        truncation_branch_values.append(pickle.load(open(DIRECTORY / truncation_run, "rb")))

    for show_individual_branches in [False, True]:
        for operator in ["〈σx〉", "〈σx σx〉", "〈σxA 4 σxB〉"]:
            max_t = max(branch_values.df_combined_values["time"])

            # Get the analytic results if we have them
            if operator in OP_NAMES_ANALYTIC and operator in DISTANCES:
                filename = (
                    OP_NAMES_ANALYTIC[operator]
                    + (f"_distance_{DISTANCES[operator]}" if DISTANCES[operator] > 0 else "")
                    + "_L_*.npy"
                )
                analytic_files = natsorted(glob.glob(str(DIRECTORY / f"exact/results/{filename}")))[
                    1:
                ]
                analytic_results_extrapolated = extrapolate_analytic_results(analytic_files)
                analytic_results_extrapolated = analytic_results_extrapolated[
                    analytic_results_extrapolated[:, 0] <= max_t
                ]

            sns.set_style("whitegrid")
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
                    "font.size": 13,
                    "figure.figsize": (10, 6.4 * 3) if show_individual_branches else (10, 6),
                }
            )

            FONTSIZE_LARGE = 17
            plt.figure()
            plt.plot(
                branch_values.df_combined_values["time"],
                branch_values.df_combined_values[operator],
                label=best_run_name,
            )

            # Plot the truncation runs
            truncation_palette = sns.color_palette("YlOrBr", n_colors=len(truncation_runs) + 1)[1:]
            for i, truncation_branch_value in enumerate(truncation_branch_values):
                plt.plot(
                    truncation_branch_value.df_combined_values["time"],
                    truncation_branch_value.df_combined_values[operator],
                    label=f"Non-branching: truncation to bond dimension {int(max(truncation_branch_value.df_combined_values['max bond dimension']))}",
                    c=truncation_palette[i],
                )

            # Plot exact results
            plt.plot(
                analytic_results_extrapolated[:, 0],
                analytic_results_extrapolated[:, 1],
                label="Exact",
                c="#0F0",
                linestyle=":",
            )

            if show_individual_branches:
                # Plot the branch values as a scatter plot
                c = "#63c7ff"
                df_branch_values = branch_values.df_branch_values
                # Sort by prob
                # df_branch_values = df_branch_values.sort_values(by='prob')
                sns.scatterplot(
                    y=np.real(df_branch_values[operator]),
                    x=df_branch_values["time"],
                    c=c,  # [(1-(p+0.1))*np.array([207, 237, 255])/255 + (p+0.1)*np.array([110, 200, 255])/255 for p in df_branch_values['prob']],
                    # size=np.abs(df_branch_values["prob"]),
                    alpha=(0.5 + df_branch_values["prob"] * 0.4),
                    s=10,
                    label="Individual branches",
                )

            ylabel = OP_NAMES_LATEX[operator]
            plt.ylabel(ylabel, fontsize=FONTSIZE_LARGE)
            plt.xlabel("Time", fontsize=FONTSIZE_LARGE)

            if not show_individual_branches:
                if operator in ["〈σx〉"]:
                    plt.ylim(0.86, 0.91)
                elif operator in ["〈σx σx〉"]:
                    plt.ylim(0.799, 0.85)
                elif operator in ["〈σx 1 σx〉", "〈σxA 2 σxB〉", "〈σx 3 σx〉", "〈σxA 4 σxB〉"]:
                    plt.ylim(0.75, 0.82)

            plt.xlim(None, 16.2)
            plt.legend()

            save_name = f"plots/best_vs_truncation_{OP_NAMES_ANALYTIC[operator]}" + (
                f"_distance_{DISTANCES[operator]}" if DISTANCES[operator] > 0 else ""
            )
            save_name = save_name + (
                "_with_individual_branches" if show_individual_branches else ""
            )
            save_name += save_suffix
            plt.savefig(save_name + ".pdf", bbox_inches="tight")
            plt.savefig(save_name + ".png", bbox_inches="tight")
            plt.show()


# %%
