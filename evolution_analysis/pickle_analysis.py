# %%
import glob
import pathlib
import pickle
import traceback
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm.auto import tqdm

from wavefunction_branching.evolve_and_branch_finite import BranchValues

NOW = datetime.now().strftime("%Y-%m-%d")


OP_NAMES_ANALYTIC = {
    "〈σx〉": "sigma_x",
    "〈σx σx〉": "sigma_x_sigma_x",
    "〈σx 1 σx〉": "sigma_x_sigma_x",
    "〈σxA 2 σxB〉": "sigma_x_sigma_x",
    "〈σx 3 σx〉": "sigma_x_sigma_x",
    "〈σxA 4 σxB〉": "sigma_x_sigma_x",
}
DISTANCES = {
    "〈σx〉": 0,
    "〈σx σx〉": 1,
    "〈σx 1 σx〉": 2,
    "〈σxA 2 σxB〉": 3,
    "〈σx 3 σx〉": 4,
    "〈σxA 4 σxB〉": 5,
}


# Deal with errors from pickling on unix and unpicking on windows or vice-versa
@contextmanager
def set_posix_windows():
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


def get_analytic_results_together(analytic_files):
    # filename = OP_NAMES_ANALYTIC[operator] + (f'_distance_{DISTANCES[operator]}' if  DISTANCES[operator] > 0 else '') + '_L_*.npy'
    # analytic_files = natsorted(glob.glob(f'exact/results/{filename}'))
    analytic_results_together = None
    Ls = []
    for analytic_file in analytic_files:
        L = int(analytic_file.split("L_")[1].split(".")[0])
        Ls.append(L)
        analytic_results = np.load(analytic_file)
        if analytic_results_together is None:
            analytic_results_together = analytic_results
        else:
            analytic_results_together = np.concatenate(
                [analytic_results_together, analytic_results[:, 1:]], axis=1
            )
    return analytic_results_together, np.array(Ls)


def best_fit_lines(x_values, y_values):
    """Returns slope and y-intercept of the best fit line of the values"""
    ms = (np.mean(x_values) * np.mean(y_values, axis=1) - np.mean(x_values * y_values, axis=1)) / (
        np.mean(x_values) ** 2 - np.mean(x_values * x_values)
    )
    cs = np.mean(y_values, axis=1) - ms * np.mean(x_values)
    return ms, cs


def extrapolate_analytic_results(analytic_files):
    analytic_results_together, Ls = get_analytic_results_together(analytic_files)
    assert analytic_results_together is not None
    _, analytic_results_extrapolated = best_fit_lines(1.0 / Ls, analytic_results_together[:, 1:])
    return np.concatenate(
        [analytic_results_together[:, :1], np.expand_dims(analytic_results_extrapolated, axis=1)],
        axis=1,
    )


def get_first_leaf(branching_mps):
    """Get the first actual MPS with a TEBD engine, rather than an empty parent"""
    if branching_mps.children is None or len(branching_mps.children) == 0:
        return branching_mps
    else:
        return get_first_leaf(branching_mps.children[0])


if __name__ == "__main__":
    WORKSPACE_PATH = Path(__file__).parent.parent.absolute()
    search_str = str(WORKSPACE_PATH / "runs/*/pickles/*.pkl")
    PICKLES_FOLDER = str(WORKSPACE_PATH / "runs")
    print(f"Searching for files matching {search_str}:")
    dont_plot = []
    pickle_files = glob.glob(search_str)
    pickle_files = [Path(p) for p in pickle_files if "extra_terms" not in p]
    for file in pickle_files:
        print("   " + str(file))

    operators = [
        "〈σx〉",
        "〈σx σx〉",
        "〈σx 1 σx〉",
        "〈σxA 2 σxB〉",
        "〈σx 3 σx〉",
        "〈σxA 4 σxB〉",
    ]  # , '〈σy〉', '〈σz〉', '〈σz σy〉', '〈σx 1 σx〉','〈σy 1 σy〉', '〈σz 1 σz〉']

    # Keep track of errors
    error_files = []

    # Import the BranchingMPS data
    Ls = []
    t_maxs = []
    filenames = []
    max_branches = []
    max_bonds = []
    branch_ats = []
    branch_values: list[pd.DataFrame] = []
    branch_values_combined: list[pd.DataFrame] = []
    with set_posix_windows():
        print("\n\nAnalyzing pickle data:")
        for pickle_file in tqdm(pickle_files):
            filename = str(pickle_file).split("pickles")[0]

            filename = (
                filename.replace(".pkl", "")
                .replace(r"pickles/", "")
                .replace("pickles\\", "")
                .replace("runs/", "")
                .replace("runs\\", "")
                .replace(PICKLES_FOLDER + "\\", "")
                .replace(PICKLES_FOLDER + "/", "")
            )
            if filename not in dont_plot:
                try:
                    with open(pickle_file, "rb") as f:
                        branch_values_raw: BranchValues = pickle.load(f)
                except Exception as e:
                    error_files.append((str(pickle_file), "Loading error", str(e)))
                    print(f"error loading {filename}")
                    print(e)
                    print("")
                    continue
                # print('\n' + filename)
                filenames.append(filename)

                max_branches.append(int(filename.split("-n")[1].split("-")[0]))
                Ls.append(int(filename.split("-L")[1].split("-")[0]))
                max_bonds.append(int(filename.split("-chi")[1].split("-")[0]))
                branch_ats.append(int(filename.split("-at")[1].split("-")[0]))
                # print(f'    L = {Ls[-1]}')
                # print(f'    max_branches = {max_branches[-1]}')
                # print(f'    max_bonds = {max_bonds[-1]}')
                # print(f'    branch_at = {branch_ats[-1]}')

                # Combine the measurements for each of the operators
                try:
                    assert branch_values_raw.df_branch_values is not None
                    for operator in operators:
                        ys = [
                            np.array(x).item().real
                            for x in branch_values_raw.df_branch_values[operator]
                        ]
                        branch_values_raw.df_branch_values[operator] = ys
                except Exception as e:
                    error_files.append((filename, "Measurement combining error", str(e)))
                    print(f"error combining measurements from {filename}")
                    print(e)
                    print("")
                    continue

                t_maxs.append(max(branch_values_raw.df_branch_values["time"]))
                # print(f'    t_max = {t_maxs[-1]}')

                branch_values.append(
                    pd.DataFrame(
                        {
                            operator: np.array(branch_values_raw.df_branch_values[operator])
                            for operator in ["time", "prob"] + operators
                        }
                    )
                )
                # branch_values_combined.append(pd.DataFrame({operator: np.array(branching_mps.branch_values.df_combined_values[operator]) for operator in ['time'] + operators}))
                assert branch_values_raw.df_combined_values is not None
                branch_values_combined.append(branch_values_raw.df_combined_values)
    max_t = max(t_maxs)

    # Get the expectation values over time, along with the analytic values if we have them
    errors_data = defaultdict(list)
    for operator in operators:
        # Get the analytic results if we have them
        if operator in OP_NAMES_ANALYTIC and operator in DISTANCES:
            filename = (
                OP_NAMES_ANALYTIC[operator]
                + (f"_distance_{DISTANCES[operator]}" if DISTANCES[operator] > 0 else "")
                + "_L_*.npy"
            )
            print(f"filename = {filename}")
            analytic_files = natsorted(
                glob.glob(str(WORKSPACE_PATH / f"exact/results/{filename}"))
            )[1:]
            print(f"analytic_files = {analytic_files}")
            j = 0
            for analytic_file in analytic_files:
                L = int(analytic_file.split("L_")[1].split(".")[0])
                analytic_results = np.load(analytic_file)
                final_point = np.sum(analytic_results[:, 0] < max_t)
                j += 1

            analytic_results_extrapolated = extrapolate_analytic_results(analytic_files)
            print(f"analytic_results_extrapolated = \n{analytic_results_extrapolated}")
            final_point = np.sum(analytic_results_extrapolated[:, 0] < max_t)
            print(f"final_point = {final_point}")

            search_str_safe = (
                search_str.replace("/", "__")
                .replace("\\", "__")
                .replace(".pkl", "")
                .replace(".", "")
                .replace("*", "#")
            )
            outfolder = WORKSPACE_PATH / f"evolution_analysis/plots/{search_str_safe}/"
            outfolder.mkdir(parents=True, exist_ok=True)

            keywords = [
                "trace_distance",
                "interference",
                "overlap",
                "prob",
                "truncation error",
                "average bond dimension",
                "average entanglement entropy",
                "one_minus_overlap",
                "tr(rho)",
            ]
            keys = set()
            for key in (branch_values_combined[-1]).keys():
                for keyword in keywords:
                    if keyword in key:
                        keys.add(key)
            print(f"keys = {keys}")

            for i in tqdm(range(len(filenames))):
                try:
                    results_numeric = np.array(branch_values_combined[i][operator])
                    times_numeric = np.round(
                        np.array(branch_values_combined[i]["time"]), decimals=3
                    )
                    times_analytic = analytic_results_extrapolated[:, 0]
                    i_numeric = 0
                    i_analytic = 0
                    times_both = []
                    results_analytic_ov = []
                    results_numeric_ov = []
                    while i_numeric < len(results_numeric) and i_analytic < len(times_analytic):
                        if times_numeric[i_numeric] > times_analytic[i_analytic]:
                            i_analytic += 1
                        elif times_numeric[i_numeric] < times_analytic[i_analytic]:
                            i_numeric += 1
                        else:
                            times_both.append(times_numeric[i_numeric])
                            results_analytic_ov.append(analytic_results_extrapolated[i_analytic, 1])
                            results_numeric_ov.append(results_numeric[i_numeric])
                            errors_data["filename"].append(filenames[i])
                            errors_data["operator"].append(operator)
                            errors_data["time"].append(times_numeric[i_numeric])
                            errors_data["L"].append(Ls[i])
                            errors_data["t_max"].append(t_maxs[i])
                            errors_data["max_branches"].append(max_branches[i])
                            errors_data["max_bonds"].append(max_bonds[i])
                            errors_data["branch_at"].append(branch_ats[i])
                            errors_data["expectation_value_analytic"].append(
                                analytic_results_extrapolated[i_analytic, 1]
                            )
                            errors_data["expectation_value_numeric"].append(
                                results_numeric[i_numeric]
                            )
                            errors_data["expectation_value_error"].append(
                                abs(
                                    results_numeric[i_numeric]
                                    - analytic_results_extrapolated[i_analytic, 1]
                                )
                            )
                            branch_values_now = branch_values[i][
                                np.round(branch_values[i]["time"], decimals=3)
                                == times_numeric[i_numeric]
                            ]
                            errors_data["n_branches"].append(len(branch_values_now))
                            errors_data["total_prob"].append(np.sum(branch_values_now["prob"]))
                            errors_data["expectation_value_standard_deviation"].append(
                                np.sqrt(
                                    np.sum(
                                        (
                                            np.array(branch_values_now["prob"])
                                            / np.sum(branch_values_now["prob"])
                                        )
                                        * (
                                            np.array(branch_values_now[operator])
                                            - results_numeric[i_numeric]
                                        )
                                        ** 2
                                    )
                                )
                            )
                            branch_values_combined_now = branch_values_combined[i][
                                np.round(branch_values_combined[i]["time"], decimals=3)
                                == times_numeric[i_numeric]
                            ]
                            errors_data["method"].append(filenames[i].split("-")[3])
                            for key in keys:
                                if key in branch_values_combined_now.keys():
                                    errors_data[key].append(branch_values_combined_now[key].item())
                                else:
                                    errors_data[key].append(0.0)
                            i_numeric += 1
                            i_analytic += 1
                except Exception as e:
                    error_files.append((filenames[i], "Error collating errors", str(e)))
                    print(f"error in {filename} when collating errors")
                    print(e)
                    print(traceback.format_exc())
                    continue

            errors_file = WORKSPACE_PATH / f"evolution_analysis/data/{NOW}/errors.pkl"
            errors_file.parent.mkdir(parents=True, exist_ok=True)
            min_len = 1e10
            for key in errors_data.keys():
                print(f"{key} = {len(errors_data[key])}")
                min_len = min(min_len, len(errors_data[key]))
            pd.DataFrame(errors_data).to_pickle(errors_file)

    # Print summary of errors
    if error_files:
        print("\nSummary of errors encountered:")
        print("-" * 80)
        for filename, error_type, error_msg in error_files:
            print(f"File: {filename}")
            print(f"Error type: {error_type}")
            print(f"Error message: {error_msg}")
            print("-" * 80)
    else:
        print("\nNo errors encountered during processing.")

# %%
