# %%
import copy
import glob
import pickle
import sys
import time
import traceback
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Literal

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tenpy
import tenpy.linalg.np_conserved as npc
import wandb
from jaxtyping import Complex
from natsort import natsorted
from opt_einops import einsum, rearrange
from tenpy.algorithms import tebd
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution

import wavefunction_branching.measure as measure
from wavefunction_branching.decompositions.decompositions import branch
from wavefunction_branching.hamiltonians import TFIChain, TFIModel
from wavefunction_branching.utils.tensors import truncate_tensor

sys.setrecursionlimit(100000)

NOW = datetime.now().strftime("%Y-%m-%d")
DEFAULT_OUT_FOLDER = Path("out") / f"{NOW}"
PICKLE_DIR = DEFAULT_OUT_FOLDER / "pickles"
PICKLE_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR = DEFAULT_OUT_FOLDER / "figures"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

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

LL_type = Complex[np.ndarray, "dim_L dim_L"]
RR_type = Complex[np.ndarray, "dim_R dim_R"]
NLR_type = Complex[np.ndarray, "N_matrices dim_L dim_R"]


def random_round(x: float) -> int:
    """Randomly round a number up or down, probabalistically to the closest integer"""
    ceil = np.ceil(x)
    floor = np.floor(x)
    p = x - floor
    r = np.random.random()
    return int(floor) if r > p else int(ceil)


def get_analytic_results_together(analytic_files: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
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
    assert isinstance(analytic_results_together, np.ndarray)
    return analytic_results_together, np.array(Ls)


def best_fit_lines(x_values, y_values):
    """Returns slope and y-intercept of the best fit line of the values"""
    ms = (np.mean(x_values) * np.mean(y_values, axis=1) - np.mean(x_values * y_values, axis=1)) / (
        np.mean(x_values) ** 2 - np.mean(x_values * x_values)
    )
    cs = np.mean(y_values, axis=1) - ms * np.mean(x_values)
    return ms, cs


def extrapolate_analytic_results(operator):
    analytic_results_together, Ls = get_analytic_results_together(operator)
    _, analytic_results_extrapolated = best_fit_lines(1.0 / Ls, analytic_results_together[:, 1:])
    return np.concatenate(
        [analytic_results_together[:, :1], np.expand_dims(analytic_results_extrapolated, axis=1)],
        axis=1,
    )


class BranchValues:
    """A class for storing expectation values of branches over time, and combining them for an estimate of the true expectation values."""

    def __init__(self):
        self.branch_values: list[dict] = []  # This is what gets updated when we add measurements
        self.df_branch_values: pd.DataFrame | None = None  # A dataframe computed from branch_values
        self.df_combined_values: pd.DataFrame | None = None  # prob-weigted average measurments

    def add_measurements_tebd(self, engine, extra_measurements=None, **kwargs):
        """Update self.branch_values with measurements from a TEBD engine (measures strictly more than add_measurements)"""
        measurements = measure.measure_tebd(engine, **kwargs)
        if extra_measurements is not None:
            measurements.update(extra_measurements)
        self.branch_values.append(measurements)

    def branch_values_to_dataframe(self):
        """Populate a dataframe from self.branch_values"""
        if self.df_branch_values is None or len(self.df_branch_values) < len(self.branch_values):
            self.df_branch_values = pd.DataFrame.from_records(self.branch_values)
            return self.df_branch_values

    def normalize_measurements(self):
        self.branch_values_to_dataframe()
        assert self.df_branch_values is not None
        columns_to_normalize = [
            column
            for column in self.df_branch_values.columns
            if pd.api.types.is_numeric_dtype(self.df_branch_values[column]) and column != "time"
        ]
        assert self.df_branch_values["time"] is not None
        for t in self.df_branch_values["time"].unique():
            for col in columns_to_normalize:
                self.df_branch_values[self.df_branch_values["time"] == t][col] = np.real(
                    self.df_branch_values[self.df_branch_values["time"] == t][col]
                    / np.sum(self.df_branch_values[self.df_branch_values["time"] == t]["prob"])
                )
        return self.df_branch_values

    def combine_measurements(self):
        """Find the prob-weighted average of each relevant column in self.branch_values, and store in self.df_combined_values"""
        self.branch_values_to_dataframe()
        assert self.df_branch_values is not None
        combined = defaultdict(list)
        columns_to_combine = [
            column
            for column in self.df_branch_values.columns
            if pd.api.types.is_numeric_dtype(self.df_branch_values[column])
            and column != "time"
            and column != "prob"
        ]
        for t in self.df_branch_values["time"].unique():
            combined["time"].append(t)
            df_now = self.df_branch_values[self.df_branch_values["time"] == t]
            combined["number of branches"].append(len(df_now))
            for col in columns_to_combine:
                combined_column_now = np.real(
                    np.sum(df_now[col] * df_now["prob"]) / np.sum(df_now["prob"])
                )
                combined[col].append(combined_column_now)
            combined["prob"].append(np.real(np.sum(df_now["prob"])))
        self.df_combined_values = pd.DataFrame(combined)
        return self.df_combined_values

    def merge_with_other(self, other):
        self.branch_values += other.branch_values
        self.branch_values_to_dataframe()
        self.combine_measurements()
        return self

    def __repr__(self):
        self.branch_values_to_dataframe()
        return self.df_branch_values.__repr__()

    def _repr_html_(self):
        self.branch_values_to_dataframe()
        return self.df_branch_values._repr_html_()  # type: ignore


def bring_into_theta_form(theta, formL, formR, sL, sR):
    return einsum((sL) ** (1.0 - formL), theta, (sR) ** (1.0 - formR), "l, b p l r, r -> b p l r")


def truncate_theta_max_bond_dims_L_M_R(theta, max_branch_bond_dims):
    theta_2site = rearrange(theta, "(pa pb) l r -> pa pb l r", pa=int(np.sqrt(theta.shape[0])))
    theta_2site, _ = truncate_tensor(
        theta_2site, "pa pb l r -> l (pa pb r)", chi_max=max_branch_bond_dims[0]
    )
    theta_2site, _ = truncate_tensor(
        theta_2site, "pa pb l r -> (pa l) (pb r)", chi_max=max_branch_bond_dims[1]
    )
    theta_2site, _ = truncate_tensor(
        theta_2site, "pa pb l r -> (pa pb l) r", chi_max=max_branch_bond_dims[2]
    )
    return rearrange(theta_2site, "pa pb l r -> (pa pb) l r")


@dataclass
class BranchingMPSConfig:
    branching: bool = True
    max_branches: int = 8
    max_branching_attempts: int | None = None
    chi_to_branch: int = 100
    steps_per_output: int = 5
    t_evo: float = 4.0
    min_relative_norm: float = 0.05
    max_trace_distance: float = 0.01
    max_overlap_error: float = 1e-5
    min_time_between_branching_attempts: float = 0.1
    discard_branch_prob_factor: float = 1e-2  # branches whose probs are less than this factor times the trace distance will be discarded
    synchronization_time: float = 1.0
    save_full_state: bool = False
    necessary_local_truncation_improvement_factor: float = 1.0
    necessary_global_truncation_improvement_factor: float = 1.0


def check_canonical_form(psi):
    norm_error = psi.norm_test()
    print("norm_error = ")
    print(norm_error)
    plt.plot(norm_error[:, 0], marker="<", label="Right canonical form error")
    plt.plot(norm_error[:, 1], marker=">", label="Left canonical form error")
    plt.ylabel("Canonical form error")
    plt.xlabel("site")
    plt.xticks(np.arange(len(norm_error)))
    plt.legend()
    plt.show()
    # assert np.allclose(norm_error, 0.0)


def set_theta(coarsegrain_from, coarsegrain_size, psi, theta, norm, trunc_params):
    """Set part of a wavefuncion psi with a new tensor"""
    old_norm = psi.norm
    assert coarsegrain_size == 2, (
        "set_svd_theta only works for coarsegrain_size=2. I should write a more general function for this."
    )
    norm_theta = np.sqrt(einsum(theta, np.conj(theta), "p l r, p l r -> "))
    theta = theta / norm_theta
    theta = rearrange(theta, "(pa pb) l r -> pa pb l r", pa=int(np.sqrt(theta.shape[0])))
    theta_npc = npc.Array.from_ndarray_trivial(theta, labels=["p0", "p1", "vL", "vR"])
    theta_npc = theta_npc.combine_legs(
        [["vL", "p0"], ["p1", "vR"]], new_axes=[0, 1], qconj=[+1, -1]
    )
    psi.set_svd_theta(coarsegrain_from, theta_npc, trunc_par=trunc_params)
    psi.canonical_form()
    psi.norm = norm * old_norm
    return psi


class BranchingMPS:
    def __init__(
        self,
        tebd_engine: tenpy.TEBDEngine
        | ExpMPOEvolution,  # The TEBD engine to use for time evolution
        cfg: BranchingMPSConfig,  # The configuration for splitting the wavefunction into branches
        branch_values: BranchValues
        | None = None,  # The structure for storing the measurements of all the branches over time
        branch_function: Callable
        | None = None,  # The function to use for finding branch decompositions in a wavefunction
        parent=None,  # a BrancingMPS from which we split (or None)
        children=None,  # a list of BrancingMPS which we have split into (or [])
        max_children=None,
        pickle_file=None,
        outfolder=DEFAULT_OUT_FOLDER,
        branching_attempts=0,
        n_times_saved=0,
        ID="",
        name="",
        wandb_project: str | None = None,
        info={},
    ):
        self.tebd_engine = tebd_engine  # The TEBD engine to use for time evolution
        self.norm = self.tebd_engine.psi.norm
        self.cfg = cfg  # The configuration for splitting the wavefunction into branches
        self.pickle_file = pickle_file
        self.outfolder = outfolder
        self.branch_function = branch_function
        self.created_walltime = datetime.now()
        self.n_times_saved = n_times_saved
        # The structure for storing the measurements of all the branches over time
        if branch_values is None:
            self.branch_values = BranchValues()
        else:
            self.branch_values = branch_values

        # a BrancingMPS from which we split (or None)
        self.parent = parent

        # a list of BrancingMPS which we have split into (or [])
        if children is None:
            self.children = []
        else:
            self.children = children

        if max_children is None:
            self.max_children = self.cfg.max_branches
        else:
            self.max_children = max_children

        self.name = name
        self.ID = ID
        print(f"Starting name = {self.name}, ID = {self.ID}")

        self.trunc_err = tenpy.linalg.truncation.TruncationError(eps=0.0, ov=1.0)
        self.dt = tebd_engine.options["dt"] if "dt" in tebd_engine.options else 1

        if self.parent is None:
            self.costFun_LM_MR_trace_distance = 0.0
            self.global_reconstruction_error_trace_distance = 0.0
            self.t_last_attempted_branching_sites = np.zeros(len(self.tebd_engine.psi.chi))
            self.site_last_attempted_branching: int | None = None
            self.last_attempted_branching_trunc_bond_dims_sites: list[
                None | tuple[int, int, int]
            ] = [None] * len(self.tebd_engine.psi.chi)
            self.last_attempted_branching_trunc_trace_distance_sites: list[None | float] = [
                None
            ] * len(self.tebd_engine.psi.chi)
            self.trace_distances = {}
            self.trace_distances["estimated_interference_error"] = 0.0
            self.trace_distances["global_reconstruction_error_trace_distance"] = (
                self.global_reconstruction_error_trace_distance
            )
            self.trace_distances["costFun_LM_MR_trace_distance"] = self.costFun_LM_MR_trace_distance
            self.time_of_last_synchronization = 0.0
            self.synchronized = False
            self.depth = 0
            self.branching_attempts = 0
            self.wandb_project = wandb_project
            if self.wandb_project is not None:
                self.run = wandb.init(
                    # Set the project where this run will be logged
                    project=self.wandb_project,
                    # Track hyperparameters and run metadata
                    config={
                        "ID": self.ID,
                        "name": self.name,
                        "max_children": self.max_children,
                        "dt": self.dt,
                        "pickle_file": self.pickle_file,
                        "outfolder": self.outfolder,
                        "created_walltime": self.created_walltime,
                        **info,
                        **asdict(cfg),
                    },
                    # Set the name of the run
                    name=self.name,
                )
            self.times_plotted = 0
            self.time_last_plotted = 0.0
        else:
            self.costFun_LM_MR_trace_distance = self.parent.costFun_LM_MR_trace_distance
            self.global_reconstruction_error_trace_distance = (
                self.parent.global_reconstruction_error_trace_distance
            )
            self.t_last_attempted_branching_sites = self.parent.t_last_attempted_branching_sites
            self.site_last_attempted_branching = self.parent.site_last_attempted_branching
            self.last_attempted_branching_trunc_bond_dims_sites = (
                self.parent.last_attempted_branching_trunc_bond_dims_sites
            )
            self.last_attempted_branching_trunc_trace_distance_sites = (
                self.parent.last_attempted_branching_trunc_trace_distance_sites
            )
            self.trace_distances = self.parent.trace_distances
            self.time_of_last_synchronization = self.parent.time_of_last_synchronization
            self.synchronized = self.parent.synchronized
            self.depth = self.parent.depth + 1
            self.branching_attempts = self.parent.branching_attempts

        self.evolved_time = float(abs(self.tebd_engine.evolved_time))
        self.t_last_attempted_branching = self.evolved_time
        self.finished = False

    def branch_and_sample(
        self,
        formL=1.0,
        formR=1.0,
        coarsegrain_from: int | Literal["half"] = "half",
        coarsegrain_size=2,
        **kwargs,
    ):
        assert self.tebd_engine is not None, (
            f"tebd_engine is None but self.children = {self.children}"
        )
        self.tebd_engine.psi.canonical_form(renormalize=False)
        # Decompose the coarsegrained region into branches
        if coarsegrain_from == "half":
            coarsegrain_from = int(self.tebd_engine.psi.L / 2 - coarsegrain_size / 2)
        assert self.branch_function is not None
        try:
            theta_purified, info = self.branch_function(
                self.tebd_engine.psi,
                formL_target=formL,
                formR_target=formR,
                coarsegrain_from=coarsegrain_from,
                coarsegrain_size=coarsegrain_size,
                **kwargs,
            )
        except Exception as e:
            print(f"ERROR in {self.ID}branch_function: \n{e}\nContinuing by ignoring it.")
            print(traceback.format_exc())
            return
        if theta_purified is None:
            return

        sL = self.tebd_engine.psi.get_SL(coarsegrain_from)
        sR = self.tebd_engine.psi.get_SR(coarsegrain_from + coarsegrain_size - 1)
        theta_purified = theta_purified[:, :, : len(sL), : len(sR)]

        # Bring into a standard canonical form where the left and right environments are identities
        # (so that the orthogonality center is in the coarsegrained region)
        if formL != 1.0 or formR != 1.0:
            theta_purified = bring_into_theta_form(theta_purified, formL, formR, sL, sR)
        # total_prob = einsum(theta_purified, np.conj(theta_purified), 'b1 p l r, b2 p l r -> ')
        # theta_purified /= total_prob**0.5

        # Measure the quality of the branch decomposition (non-interference and faithfulness)

        # density_matrix_terms = einsum(
        #     theta_purified, np.conj(theta_purified),
        #     'b  p  l  r ,            bc pc l r      ->  b bc p pc'
        # )
        # overlaps_between_branches = einsum(density_matrix_terms, 'b bc p p -> b bc')

        branch_probs = np.abs(
            einsum(theta_purified, np.conj(theta_purified), " b p l r, b p l r -> b")
        )
        total_prob = branch_probs.sum()
        print(f"{self.ID}total_prob: {total_prob}")
        branch_probs = branch_probs / total_prob
        print(f"{self.ID}branch_probs: {branch_probs}")
        theta_purified /= np.sqrt(total_prob)
        # print(f'{self.ID}overlaps: {overlaps_between_branches}')

        # Quantify quality of decomposition, don't branch if it's too low
        theta_orig = rearrange(
            self.tebd_engine.psi.get_theta(
                coarsegrain_from, n=coarsegrain_size, cutoff=0.0, formL=1.0, formR=1.0
            ).to_ndarray(),
            "L p1 p2 R -> (p1 p2) L R",
        )
        # norm_orig = np.sqrt(np.abs(einsum(theta_orig, np.conj(theta_orig), 'p l r, p l r -> ')))

        trace_distances = measure.LMR_trace_distances(
            theta_orig, theta_purified, measure_split=True
        )
        costFun_LM_MR_trace_distance = 0.5 * (
            trace_distances["trace_distance_LM"] + trace_distances["trace_distance_MR"]
        )
        trace_distances["costFun_LM_MR_trace_distance"] = costFun_LM_MR_trace_distance
        trace_distances["costFun_split_trace_distance"] = (
            trace_distances["trace_distance_Lpl"]
            + trace_distances["trace_distance_M"]
            + trace_distances["trace_distance_Rpr"]
        ) / 3.0
        # print(f'{self.ID}trace_distances:')
        # for key, value in trace_distances.items():
        #     print(f'    {key}: {value}')

        # overlaps_with_orig = abs(einsum(theta_purified[..., :theta_orig.shape[-2], :theta_orig.shape[-1]], np.conj(theta_orig)[..., :theta_purified.shape[-2], :theta_purified.shape[-1]], 'b p l r, p l r -> b'))

        # print(f'{self.ID}overlaps_with_orig: {overlaps_with_orig}')

        global_reconstruction_error_trace_distance = trace_distances[
            "global_reconstruction_error_trace_distance"
        ]
        # print(f'{self.ID}global_reconstruction_error_trace_distance: {global_reconstruction_error_trace_distance}')

        # if len(info) > 0:
        #     print('decomposition info:')
        #     for key, value in info.items():
        #         print(f'    {key}: {value}')

        if (
            costFun_LM_MR_trace_distance > self.cfg.max_trace_distance
            or global_reconstruction_error_trace_distance > self.cfg.max_overlap_error
        ):
            print(
                f"{self.ID}Rejecting the branch decomposition: \n    costFun_LM_MR_trace_distance = {costFun_LM_MR_trace_distance}, max_trace_distance = {self.cfg.max_trace_distance} \n    global_reconstruction_error_trace_distance = {global_reconstruction_error_trace_distance}, max_overlap_error = {self.cfg.max_overlap_error} "
            )
            return
        elif "rejected" in info.keys():
            if info["rejected"]:
                print(
                    f"{self.ID}Rejecting the branch decomposition because info['rejected'] is True (returned from self.branch_function = {self.branch_function})"
                )
                return
        else:
            print(
                f"{self.ID}costFun_LM_MR_trace_distance = {costFun_LM_MR_trace_distance}, max_trace_distance = {self.cfg.max_trace_distance} \n    global_reconstruction_error_trace_distance = {global_reconstruction_error_trace_distance}, max_overlap_error = {self.cfg.max_overlap_error} "
            )

        self.costFun_LM_MR_trace_distance += costFun_LM_MR_trace_distance
        self.global_reconstruction_error_trace_distance += (
            global_reconstruction_error_trace_distance
        )
        print(f"{self.ID}total trace norm decomposition_error: {self.costFun_LM_MR_trace_distance}")
        print(
            f"{self.ID}total overlap decomposition_error: {self.global_reconstruction_error_trace_distance}"
        )

        # Sample from the branches

        # Filter out branches if the number of branches is more than our max_children
        branch_indices = np.random.choice(
            np.arange(len(branch_probs)),
            p=branch_probs / branch_probs.sum(),
            replace=False,
            size=min((branch_probs > 0).sum(), self.max_children),
        )
        print(f"{self.ID}branch_indices = {branch_indices}")

        # Assign the number of max children to each child branch, proportional to their probs (randomly)
        selected_branch_probs = np.array([branch_probs[i] for i in branch_indices])
        selected_branch_probs_rescaled = selected_branch_probs / np.sum(selected_branch_probs)
        # TODO: MAKE THIS MORE EFFICIENT
        i = 0
        max_children = [-999]
        while np.sum(max_children) != self.max_children and i < 10000:
            max_children = [
                max(0, random_round(self.max_children * selected_branch_probs_rescaled[i]))
                for i in range(len(selected_branch_probs_rescaled))
            ]
            if i >= 10000:
                print(f"No good assignment of max_children found - max_children = {max_children}")
                break
        # for i in range(len(branch_indices)):
        #     if max_children[i] > 0:
        #         # Drop any branches with a prob smaller than the error we're already introducing in the decomposition
        #         if selected_branch_probs_rescaled[i] < costFun_LM_MR_trace_distance*self.cfg.discard_branch_prob_factor:
        #             max_children[i] = 0
        #             print(f'{self.ID}Discarding branch {i} with prob {selected_branch_probs_rescaled[i]:.2E} < costFun_LM_MR_trace_distance*self.cfg.discard_branch_prob_factor = {costFun_LM_MR_trace_distance:.2E}*{self.cfg.discard_branch_prob_factor:.2E} = {costFun_LM_MR_trace_distance * self.cfg.discard_branch_prob_factor:.2E}')
        print(
            f"{self.ID}self.max_children = {self.max_children} sum(children max_children) = {sum(max_children)} children max_children = {max_children}"
        )
        branch_indices = [
            branch_indices[i] for i in range(len(branch_indices)) if max_children[i] > 0
        ]
        print(f"{self.ID}branch_indices with nonzero max_children = {branch_indices}")

        # branch_psis = []
        # for i in range(len(branch_indices)):
        #     prob = branch_probs[branch_indices[i]]
        #     theta = theta_purified[branch_indices[i]] #if len(branch_indices) > 1 else theta_purified[branch_indices[i]] / np.sqrt(prob)
        # tebd_engine = copy.deepcopy(self.tebd_engine)
        # branch_psis.append(set_theta(coarsegrain_from, coarsegrain_size, tebd_engine.psi, theta, prob, tebd_engine.trunc_params))

        trace_distances_with_sampling = measure.LMR_trace_distances(
            theta_orig, theta_purified[branch_indices]
        )
        # trace_distance_with_sampling = 0.5 * (trace_distances_with_sampling['trace_distance_LM'] + trace_distances_with_sampling['trace_distance_MR'])
        print(f"{self.ID}trace_distances_with_sampling:")
        for key, value in trace_distances_with_sampling.items():
            print(f"    {key}: {value}")

        # Truncate the bond dimensions of the branches
        thetas_truncated = []
        bond_dims_L = []
        bond_dims_M = []
        bond_dims_R = []
        trunc_kwargs = {
            "svd_min": self.tebd_engine.trunc_params["svd_min"],
            "trunc_cut": self.tebd_engine.trunc_params["trunc_cut"],
            "chi_max": self.tebd_engine.trunc_params["chi_max"],
        }
        for i in range(len(branch_indices)):
            theta = theta_purified[branch_indices[i]]
            try:
                theta_2site = rearrange(
                    theta, "(pa pb) l r -> pa pb l r", pa=int(np.sqrt(theta.shape[0]))
                )
                theta_2site, bond_dim_L = truncate_tensor(
                    theta_2site, "pa pb l r -> l (pa pb r)", **trunc_kwargs
                )
                theta_2site, bond_dim_M = truncate_tensor(
                    theta_2site, "pa pb l r -> (pa l) (pb r)", **trunc_kwargs
                )
                theta_2site, bond_dim_R = truncate_tensor(
                    theta_2site, "pa pb l r -> (pa pb l) r", **trunc_kwargs
                )
                thetas_truncated.append(rearrange(theta_2site, "pa pb l r -> (pa pb) l r"))
                bond_dims_L.append(bond_dim_L)
                bond_dims_M.append(bond_dim_M)
                bond_dims_R.append(bond_dim_R)
            except np.linalg.LinAlgError as e:
                print(f"np.linalg.LinAlgError in truncation of branch bond dimensions: {e}")
                print(traceback.format_exc())
                thetas_truncated.append(theta)
                bond_dims_L.append(theta.shape[-2])
                bond_dims_R.append(theta.shape[-1])
                bond_dims_M.append(
                    int(np.sqrt(theta.shape[0]) * min((bond_dims_L[-1], bond_dims_R[-1])))
                )
        print(f"{self.ID}bond_dims_L = {bond_dims_L}")
        print(f"{self.ID}bond_dims_M = {bond_dims_M}")
        print(f"{self.ID}bond_dims_R = {bond_dims_R}")
        max_bond_L = max(bond_dims_L)
        max_bond_M = max(bond_dims_M)
        max_bond_R = max(bond_dims_R)
        max_branch_bond_dims = (max_bond_L, max_bond_M, max_bond_R)

        trace_distances_with_sampling_and_truncation = measure.LMR_trace_distances(
            theta_orig, np.stack(thetas_truncated)
        )
        trace_distance_with_sampling_and_truncation = 0.5 * (
            trace_distances_with_sampling_and_truncation["trace_distance_LM"]
            + trace_distances_with_sampling_and_truncation["trace_distance_MR"]
        )
        print(f"{self.ID}trace_distance_with_sampling_and_truncation:")
        for key, value in trace_distances_with_sampling_and_truncation.items():
            print(f"    {key}: {value}")

        # Quantify the errors which would arise from truncating the original state to equivalent bond dimensions
        theta_orig_trunc = truncate_theta_max_bond_dims_L_M_R(theta_orig, max_branch_bond_dims)
        trace_distances_truncation_only_comparison = measure.LMR_trace_distances(
            theta_orig, np.expand_dims(theta_orig_trunc, axis=0)
        )
        trace_distance_truncation_only_comparison = 0.5 * (
            trace_distances_truncation_only_comparison["trace_distance_LM"]
            + trace_distances_truncation_only_comparison["trace_distance_MR"]
        )
        global_reconstruction_error_truncation_only_comparison = (
            trace_distances_truncation_only_comparison["global_reconstruction_error_trace_distance"]
        )
        print(f"{self.ID}trace_distances_truncation_only_comparison:")
        for key, value in trace_distances_truncation_only_comparison.items():
            print(f"    {key}: {value}")

        # If the error from the branch decomposition is larger than what you'd get from truncation, don't branch
        bad_local_trace_distance = (
            costFun_LM_MR_trace_distance * self.cfg.necessary_local_truncation_improvement_factor
            > trace_distance_truncation_only_comparison
        )
        bad_global_trace_distance = (
            global_reconstruction_error_trace_distance
            * self.cfg.necessary_global_truncation_improvement_factor
            > global_reconstruction_error_truncation_only_comparison
        )
        if bad_local_trace_distance or bad_global_trace_distance:
            print(
                f"{self.ID}Rejecting the branch decomposition: {'bad local trace distance' if bad_local_trace_distance else 'bad global trace distance'}"
            )
            print(
                f"    local_trace_distance = {costFun_LM_MR_trace_distance:.2e} | truncation_only_comparison = {trace_distance_truncation_only_comparison:.2e} - ratio = {(trace_distance_truncation_only_comparison / costFun_LM_MR_trace_distance):2e} (necessary: {self.cfg.necessary_local_truncation_improvement_factor})"
            )
            print(
                f"    global_trace_distance = {global_reconstruction_error_trace_distance:.2e} | truncation_only_comparison = {global_reconstruction_error_truncation_only_comparison:.2e} - ratio = {(global_reconstruction_error_truncation_only_comparison / global_reconstruction_error_trace_distance):2e} (necessary: {self.cfg.necessary_global_truncation_improvement_factor})"
            )
            self.last_attempted_branching_trunc_bond_dims_sites[coarsegrain_from] = (
                max_branch_bond_dims
            )
            self.last_attempted_branching_trunc_trace_distance_sites[coarsegrain_from] = (
                trace_distance_truncation_only_comparison
            )
            return
        else:
            print(f"{self.ID}Accepting the branch decomposition: ")
            print(
                f"    local_trace_distance = {costFun_LM_MR_trace_distance:.2e} < truncation_only_comparison = {trace_distance_truncation_only_comparison:.2e}"
            )
            print(
                f"    global_trace_distance = {global_reconstruction_error_trace_distance:.2e} < truncation_only_comparison = {global_reconstruction_error_truncation_only_comparison:.2e}"
            )
            self.last_attempted_branching_trunc_bond_dims_sites[coarsegrain_from] = None
            self.last_attempted_branching_trunc_trace_distance_sites[coarsegrain_from] = None

        # If we've made it this far, we're going to branch

        for key, value in trace_distances.items():
            if key not in self.trace_distances:
                self.trace_distances[key] = 0.0
            self.trace_distances[key] += value

        for key, value in trace_distances_with_sampling.items():
            if key + "_with_sampling" not in self.trace_distances:
                self.trace_distances[key + "_with_sampling"] = 0.0
            self.trace_distances[key + "_with_sampling"] += value

        for key, value in trace_distances_with_sampling_and_truncation.items():
            if key + "_with_sampling_and_truncation" not in self.trace_distances:
                self.trace_distances[key + "_with_sampling_and_truncation"] = 0.0
            self.trace_distances[key + "_with_sampling_and_truncation"] += value

        for key, value in trace_distances_truncation_only_comparison.items():
            if key + "_truncation_only_comparison" not in self.trace_distances:
                self.trace_distances[key + "_truncation_only_comparison"] = 0.0
            self.trace_distances[key + "_truncation_only_comparison"] += value

        if len(branch_indices) == 0:
            # branch_indices = [np.argmax(branch_probs)]
            print(f"{self.ID}No further branch_indices were selected: len(branch_indices) = 0")
            print(f"{self.ID}TERMINATING.")
            self.finished = True
            self.branch_values.add_measurements_tebd(
                self.tebd_engine, extra_measurements=self.trace_distances
            )
            return

        if len(branch_indices) == 1:
            theta = thetas_truncated[0]
            prob = branch_probs[branch_indices[0]]
            self.tebd_engine.psi = set_theta(
                coarsegrain_from,
                coarsegrain_size,
                self.tebd_engine.psi,
                theta,
                np.sqrt(prob),
                self.tebd_engine.trunc_params,
            )
            self.norm = self.tebd_engine.psi.norm
            self.ID = self.ID + f"{branch_indices[0]}|"
            self.depth += 1
        elif len(branch_indices) > 1:
            for i in range(len(branch_indices)):
                theta = thetas_truncated[i]
                prob = branch_probs[branch_indices[i]]
                tebd_engine = copy.deepcopy(self.tebd_engine)
                tebd_engine.psi = set_theta(
                    coarsegrain_from,
                    coarsegrain_size,
                    tebd_engine.psi,
                    theta,
                    np.sqrt(prob),
                    tebd_engine.trunc_params,
                )
                self.children.append(
                    BranchingMPS(
                        tebd_engine,
                        self.cfg,
                        branch_values=self.branch_values,
                        branch_function=self.branch_function,
                        parent=self,
                        max_children=int(max_children[i]),
                        ID=self.ID + f"{branch_indices[i]}|",
                        n_times_saved=self.n_times_saved,
                        name=self.name,
                    )
                )

            # Sort the children from high prob to low prob
            self.children.sort(key=lambda x: np.abs(x.tebd_engine.psi.norm) ** 2, reverse=True)
            # Remove our actual MPS data if we've got children
            del self.tebd_engine

    def find_region_to_branch(self) -> tuple[int, int] | tuple[None, None]:
        assert self.tebd_engine is not None, (
            f"tebd_engine is None but self.children = {self.children}"
        )
        if self.branch_function is None or not self.cfg.branching:
            return None, None
        chis = np.array(self.tebd_engine.psi.chi, dtype=float)
        bias = 0.25 - (np.arange(len(chis)) / (len(chis) - 1) - 0.5) ** 2
        chis += bias
        last_branching_times = np.array(self.t_last_attempted_branching_sites)
        ts = (
            self.evolved_time - last_branching_times
        ) - self.cfg.min_time_between_branching_attempts / (self.tebd_engine.psi.L / 2)
        if all(ts < 0):
            return None, None
        bond_ind = np.argmax(chis * ts)
        if chis[bond_ind] < self.cfg.chi_to_branch:
            return None, None
        else:
            bond_inds_nearby = np.array(
                [bond_ind, bond_ind + 1, bond_ind - 1, bond_ind + 2, bond_ind - 2]
            )
            bond_inds_nearby = bond_inds_nearby[
                (bond_inds_nearby >= 0) & (bond_inds_nearby < len(chis))
            ]
            for ind in bond_inds_nearby:
                self.t_last_attempted_branching_sites[ind] = 0.5 * (
                    self.t_last_attempted_branching_sites[ind] + self.evolved_time
                )
            if (
                self.site_last_attempted_branching is not None
                and self.site_last_attempted_branching not in bond_inds_nearby
            ):
                bond_inds_nearby = np.append(bond_inds_nearby, self.site_last_attempted_branching)
            # Quantify the errors which would arise from truncating to equivalent bond dimensions from the last attempted branching nearby
            for ind in bond_inds_nearby:
                if self.last_attempted_branching_trunc_bond_dims_sites[ind] is not None:
                    theta_orig = rearrange(
                        self.tebd_engine.psi.get_theta(
                            ind, n=2, cutoff=0.0, formL=1.0, formR=1.0
                        ).to_ndarray(),
                        "L p1 p2 R -> (p1 p2) L R",
                    )
                    theta_orig_trunc = truncate_theta_max_bond_dims_L_M_R(
                        theta_orig, self.last_attempted_branching_trunc_bond_dims_sites[ind]
                    )
                    trace_distances_truncation_only_comparison = measure.LMR_trace_distances(
                        theta_orig, np.expand_dims(theta_orig_trunc, axis=0)
                    )
                    trace_distance_truncation_only_comparison = 0.5 * (
                        trace_distances_truncation_only_comparison["trace_distance_LM"]
                        + trace_distances_truncation_only_comparison["trace_distance_MR"]
                    )

                    spatial_decay_factor = 1.0 / (
                        ((ind - bond_ind) / (0.15 * len(chis) + 1)) ** 2 + 1
                    )
                    temporal_decay_factor = 1.0 / (
                        1
                        + (self.evolved_time - self.t_last_attempted_branching_sites[ind])
                        / self.cfg.min_time_between_branching_attempts
                    )

                    # Don't attempt branching again if the truncation error is't approaching the error from the last attempted branching
                    if (
                        trace_distance_truncation_only_comparison
                        < self.last_attempted_branching_trunc_trace_distance_sites[ind]
                        * spatial_decay_factor
                        * temporal_decay_factor
                    ):
                        print(
                            f"{self.ID}Not proceeding with branch finding at site {bond_ind} because the error from truncating to the bond dimensions of the last attempted branching at site {ind} was much smaller than the error of the last attempted branching there: \n    trace_distance_truncation_only_comparison = {trace_distance_truncation_only_comparison} < last_attempted_branching_trunc_trace_distance = {self.last_attempted_branching_trunc_trace_distance_sites[ind]} "
                        )
                        return None, None
                    else:
                        print(
                            f"{self.ID}Proceeding with branch finding at site {bond_ind} because the error from truncating to the bond dimensions of the last attempted branching at site {ind} was larger (or at least not much smaller) than the error of the last attempted branching there: \n    trace_distance_truncation_only_comparison = {trace_distance_truncation_only_comparison} vs. last_attempted_branching_trunc_trace_distance = {self.last_attempted_branching_trunc_trace_distance_sites[ind]} "
                        )
            self.t_last_attempted_branching_sites[bond_ind] = self.evolved_time
            self.site_last_attempted_branching = int(bond_ind)
            return int(bond_ind), 2

    def get_root(self):
        if self.parent is not None:
            return self.parent.get_root()
        else:
            return self

    def get_leaf(self, i):
        if self.children is None or len(self.children) == 0:
            return self
        else:
            return self.children[i % len(self.children)].get_leaf(i // len(self.children))

    def get_random_leaf(self, prob_weighted=False):
        if self.children is None or len(self.children) == 0:
            return self
        else:
            if prob_weighted:
                weights = np.array([np.abs(child.norm) ** 2 for child in self.children])
                weights /= np.sum(weights)
                i = np.random.choice(len(self.children), p=weights)
            else:
                i = np.random.randint(len(self.children))
            return self.children[i].get_random_leaf(prob_weighted=prob_weighted)

    def count_leaves(self):
        if self.children is None or len(self.children) == 0:
            return 1
        else:
            return sum([child.count_leaves() for child in self.children])

    def save(self, final=False):
        if self.parent is not None:
            self.parent.save()
        else:
            t0 = time.time()
            if self.pickle_file is not None:
                branchvals_file = str(self.pickle_file).split(".pkl")[0] + "_branch_values.pkl"
                branchvals_file = (
                    branchvals_file
                    if (self.n_times_saved % 2 == 0 or final)
                    else branchvals_file + "tmp"
                )
                with open(branchvals_file, "wb") as f:
                    pickle.dump(self.branch_values, f)
                    f.close()
                if self.cfg.save_full_state:
                    pickle_file = (
                        self.pickle_file
                        if (self.n_times_saved % 2 == 0 or final)
                        else str(self.pickle_file) + "tmp"
                    )
                    with open(pickle_file, "wb") as f:
                        pickle.dump(self, f)
                        f.close()
                self.n_times_saved += 1

                if final:
                    print(f"{self.ID}Removing temp files as this is the final save.")
                    for tmp_file in [branchvals_file + "tmp", str(self.pickle_file) + "tmp"]:
                        Path(tmp_file).unlink(missing_ok=True)
            t1 = time.time()
            print(f"{self.ID}Saved in {t1 - t0} seconds to {self.pickle_file}")

    def plot(self):
        if self.parent is not None:
            self.parent.plot()
        else:
            plots_dir = self.outfolder / "plots"
            plots_dir.mkdir(exist_ok=True, parents=True)
            self.branch_values.combine_measurements()
            df_branch_values = self.branch_values.df_branch_values
            df_combined_values = self.branch_values.df_combined_values
            assert df_branch_values is not None
            assert df_combined_values is not None

            max_t = max(df_combined_values["time"])

            sns.set_style("whitegrid")
            plt.rcParams["figure.figsize"] = [14, 16]

            # Plot expectation values over time
            for operator in [
                "〈σx〉",
                "〈σx σx〉",
                "〈σx 1 σx〉",
                "〈σxA 2 σxB〉",
                "〈σx 3 σx〉",
                "〈σxA 4 σxB〉",
                "prob",
            ]:  # , '〈σy〉', '〈σz〉', '〈σz σy〉', '〈σx 1 σx〉','〈σy 1 σy〉', '〈σz 1 σz〉']:
                plt.figure(dpi=150)
                c = sns.color_palette("YlOrBr", n_colors=1)[0]
                sns.scatterplot(
                    y=np.real(df_branch_values[operator]),
                    x=df_branch_values["time"],
                    c=c,
                    size=np.abs(df_branch_values["prob"]),
                    alpha=0.5,
                )  # , label=f'{self.name} (individual)'
                plt.plot(
                    df_combined_values["time"],
                    np.real(df_combined_values[operator]),
                    c="black",
                    label="weighted average",
                )

                # Plot the analytic results if we have them
                if operator in OP_NAMES_ANALYTIC and operator in DISTANCES:
                    filename = (
                        OP_NAMES_ANALYTIC[operator]
                        + (f"_distance_{DISTANCES[operator]}" if DISTANCES[operator] > 0 else "")
                        + "_L_*.npy"
                    )
                    print(f"filename = {filename}")
                    analytic_files = natsorted(glob.glob(f"exact/results/{filename}"))
                    analytic_palette = sns.color_palette(
                        "blend:#FF0,#0F0", n_colors=len(analytic_files) + 1
                    )
                    print(f"analytic_files = {analytic_files}")
                    j = 0
                    for analytic_file in analytic_files:
                        L = int(analytic_file.split("L_")[1].split(".")[0])
                        analytic_results = np.load(analytic_file)
                        final_point = np.sum(analytic_results[:, 0] < max_t)
                        plt.plot(
                            analytic_results[:final_point, 0],
                            analytic_results[:final_point, 1],
                            label=f"Analytic PBC, L = {L}",
                            linestyle=":",
                            c=analytic_palette[j],
                            alpha=0.4,
                        )  # alpha=1.0/len(analytic_files))
                        j += 1

                    analytic_results_extrapolated = extrapolate_analytic_results(analytic_files)
                    final_point = np.sum(analytic_results_extrapolated[:, 0] < max_t)
                    plt.plot(
                        analytic_results_extrapolated[:final_point, 0],
                        analytic_results_extrapolated[:final_point, 1],
                        label="Analytic PBC, L = infinty (extrapolated)",
                        linestyle=":",
                        c=analytic_palette[-1],
                    )

                ylabel = operator.replace("〈", "").replace("〉", "").replace("σ", "sigma_")
                plt.title(f"{ylabel} over time: {self.name}")
                plt.ylabel(ylabel)
                plt.xlim(0.0, self.cfg.t_evo)
                if operator in [
                    "〈σx〉",
                    "〈σx σx〉",
                ]:
                    plt.ylim(0.75, 1.0)
                elif operator in ["〈σx 1 σx〉", "〈σxA 2 σxB〉", "〈σx 3 σx〉", "〈σxA 4 σxB〉"]:
                    plt.ylim(0.6, 0.9)
                plt.savefig(plots_dir / f"{NOW}_{self.name}_{ylabel}.pdf")
                plt.savefig(plots_dir / f"{NOW}_{self.name}_{ylabel}.png")
                if self.wandb_project is not None:
                    wandb.log(
                        {
                            f"plots/{ylabel}": wandb.Image(
                                str(plots_dir / f"{NOW}_{self.name}_{ylabel}.png")
                            )
                        },
                        step=self.times_plotted,
                    )
                plt.show()
                plt.clf()
                plt.cla()

            # Plot error estimates over time
            LMR_palette = sns.color_palette("Paired")[2:]

            def trace_distance_colors(key):
                cm_key = key.split("_")[2]
                color_map = {
                    "LM": LMR_palette[0],
                    "MR": LMR_palette[1],
                    "L": LMR_palette[2],
                    "M": LMR_palette[3],
                    "R": LMR_palette[4],
                }
                if cm_key in color_map:
                    return color_map[cm_key]
                elif key == "costFun_LM_MR_trace_distance":
                    return LMR_palette[-1]
                elif key == "global_reconstruction_error_trace_distance":
                    return LMR_palette[-2]
                else:
                    return (0.0, 0.0, 0.0)

            plt.figure(dpi=150)
            plt.plot(
                df_combined_values["time"],
                df_combined_values["truncation error"],
                label="truncation error",
            )

            for key in self.trace_distances:
                if ("trace_distance" in key) and ("Lpl" not in key) and ("Rpr" not in key):
                    color = trace_distance_colors(key)
                    linestyle = (
                        "dashdot"
                        if "with_sampling_and_truncation" in key
                        else "dashed"
                        if "with_sampling" in key
                        else "dotted"
                        if "truncation_only_comparison" in key
                        else "solid"
                    )
                    plt.plot(
                        df_combined_values["time"],
                        df_combined_values[key],
                        label=key.replace("_", " "),
                        linestyle=linestyle,
                        color=color,
                        alpha=0.75,
                    )
                if "interference" in key:
                    if "total" in key:
                        color = sns.color_palette("Paired")[9]
                    else:
                        color = sns.color_palette("Paired")[8]
                    plt.plot(
                        df_combined_values["time"],
                        df_combined_values[key],
                        label=key.replace("_", " "),
                        color=color,
                    )
            plt.yscale("symlog", linthresh=1e-10)
            plt.xlim(0.0, self.cfg.t_evo)
            plt.legend()
            plt.title(f"Weighted average errors over time: {self.name}")
            plt.savefig(plots_dir / f"{NOW}_{self.name}_errors.pdf")
            plt.savefig(plots_dir / f"{NOW}_{self.name}_errors.png")
            if self.wandb_project is not None:
                wandb.log(
                    {"plots/errors": wandb.Image(str(plots_dir / f"{NOW}_{self.name}_errors.png"))},
                    step=self.times_plotted,
                )
            plt.show()
            plt.clf()
            plt.cla()

            # Plot the bond dimensions over time
            plt.figure(dpi=150)
            L = len(df_branch_values["bond dimensions"][0])
            palette = sns.color_palette("YlOrBr", n_colors=(L + 1) // 2)
            palette += palette[::-1]
            for i in range(L):
                sns.lineplot(
                    df_branch_values,
                    y=[x[i] for x in df_branch_values["bond dimensions"]],
                    x="time",
                    c=palette[i],
                )  # , label=f'{self.name} bond {i}')
            sns.scatterplot(
                df_branch_values, y="max bond dimension", x="time", c="orange"
            )  # , label=f'{self.name} branches')
            plt.plot(
                df_combined_values["time"],
                df_combined_values["max bond dimension"],
                c="black",
                label="weighted average max bond dimension",
            )
            plt.title(f"Bond dimensions over time: {self.name}")
            plt.xlim(0.0, self.cfg.t_evo)
            plt.savefig(plots_dir / f"{NOW}_{self.name}_bond_dimensions.pdf")
            plt.savefig(plots_dir / f"{NOW}_{self.name}_bond_dimensions.png")
            if self.wandb_project is not None:
                wandb.log(
                    {
                        "plots/bond_dimensions": wandb.Image(
                            str(plots_dir / f"{NOW}_{self.name}_bond_dimensions.png")
                        )
                    },
                    step=self.times_plotted,
                )
            plt.show()
            plt.clf()
            plt.cla()

            # Log to wandb
            if self.wandb_project is not None:
                # Select just the most recent combined values
                save_dict = df_combined_values[df_combined_values["time"] == max_t].to_dict(
                    orient="records"
                )[0]
                save_dict = {"combined/" + key: value for key, value in save_dict.items()}
                save_dict["combined/truncation error"] = save_dict["errors/truncation error"]
                # df_branch_values_recent = df_branch_values[df_branch_values['time'] > self.time_last_plotted]
                print(f"save_dict = {save_dict}")
                log_dict = {
                    "costFun_LM_MR_trace_distance": self.costFun_LM_MR_trace_distance,
                    "global_reconstruction_error_trace_distance": self.global_reconstruction_error_trace_distance,
                    "t_last_attempted_branching_sites": self.t_last_attempted_branching_sites,
                    "site_last_attempted_branching": self.site_last_attempted_branching,
                    "last_attempted_branching_trunc_bond_dims_sites": self.last_attempted_branching_trunc_bond_dims_sites,
                    "last_attempted_branching_trunc_trace_distance_sites": self.last_attempted_branching_trunc_trace_distance_sites,
                    "time_of_last_synchronization": self.time_of_last_synchronization,
                    "depth": self.depth,
                    "branching_attempts": self.branching_attempts,
                    "max_children": self.max_children,
                    "n_branches": self.count_leaves(),
                    "n_times_saved": self.n_times_saved,
                    "times_plotted": self.times_plotted,
                    # 'diagram':self.print_tree(),
                    **{"errors/" + key: value for key, value in self.trace_distances},
                    **save_dict,
                }

                # Cast complex values to float
                for key, value in log_dict.items():
                    if isinstance(value, complex):
                        log_dict[key] = float(value.real)

                print(f"log_dict = {log_dict}")
                print(f"trace_distances = {self.trace_distances}")

                wandb.log(log_dict, step=self.times_plotted)
            self.times_plotted += 1
            self.time_last_plotted = max_t

    def print_tree(self, header="", last=True, highlight_id=None):
        elbow = "└──"
        pipe = "│  "
        tee = "├──"
        blank = "   "
        outstr = header + (elbow if last else tee)
        final_id = "0" if len(self.ID) < 2 else self.ID[-2]
        finished = (
            "FINISHED    "
            if self.finished
            else "SYNCHRONIZED"
            if self.synchronized
            else "RUNNING     "
            if highlight_id == self.ID
            else "            "
        )
        highlight = "******************" if highlight_id == self.ID else ""
        if len(self.children) > 0:
            outstr += f"{final_id} (depth {self.depth}) {finished} trace norm decomposition error | {self.costFun_LM_MR_trace_distance:.2E} | overlap decomposition error = {self.global_reconstruction_error_trace_distance:.2E} {highlight}\n"
        else:
            outstr += f"{final_id} (depth {self.depth}) {finished} t = {self.evolved_time:.3f} | prob = {np.abs(self.tebd_engine.psi.norm) ** 2:.2E} | max_children = {self.max_children} {highlight}\n"
        for i, c in enumerate(self.children):
            outstr += c.print_tree(
                header=header + (blank if last else pipe),
                last=i == len(self.children) - 1,
                highlight_id=highlight_id,
            )
        return outstr

    def __repr__(self):
        return f"{self.name}:\n" + self.print_tree()

    def estimate_inteference_error(self, n_samples=50):
        error = 0.0
        if self.parent is not None:
            return self.parent.estimate_inteference_error()
        elif len(self.children) == 0:
            return error
        else:
            print("Estimating interference error")
            # Estimate the interference error by measuring trace norms of two-site-cross-terms between branches
            # This is measured for the two sites in the middle of the chain only.
            # Estimate the average interference error, by random samping over combinations of branches.
            # Then just approximate it as all pairs having that same average interference error (or plot the average interference error over time, as well as the estimated total interference error).
            # The total interference error should be estimated as sqrt(n_terms)*average_interference_error_per_term, where n_terms is (n_branches-1)^2. So it simplifies to (n_branches-1)*average_interference_error_per_term

            def partial_overlap(psi1, psi2, except_sites):
                assert except_sites[0] > 0

                p = psi2._get_p_label("")
                pc = psi1._get_p_label("*")

                B_ket = psi2.get_B(0)
                B_bra = psi1.get_B(0)
                C = npc.tensordot(B_bra.conj(), B_ket, axes=[["vL*"] + pc, ["vL"] + p])  # type: ignore
                L = psi1.L
                forms = [(0, 1)] * except_sites[0] + [(1, 1)] + [(1, 0)] * (L - except_sites[0] - 1)
                for i in range(1, L):
                    B_ket = psi2.get_B(i, forms[i])
                    C = npc.tensordot(C, B_ket, axes=["vR", "vL"])  # type: ignore
                    B_bra = psi1.get_B(i, forms[i])
                    if i in except_sites:
                        C = npc.tensordot(C, B_bra.conj(), axes=["vR*", "vL*"])  # type: ignore
                        C = C.replace_labels(
                            p + pc, [x + f"{i}" for x in p] + [x + f"{i}" for x in pc]
                        )
                    else:
                        C = npc.tensordot(C, B_bra.conj(), axes=[["vR*"] + p, ["vL*"] + pc])  # type: ignore

                C = npc.trace(C, "vR", "vR*")  # type: ignore
                C = C.transpose(
                    sorted(C.get_leg_labels())  # type: ignore
                )  # eg. ['p*15', 'p*16', 'p15', 'p16']
                return C.to_ndarray()

            n_branches = self.count_leaves()

            # Check all pairs of branches if n_branches^2 <= n_samples, and then weight by norms later
            # Otherwise do norm-weighted sampling with replacement n_branches^2 > n_samples, and don't weight by norms (to avoid double counting)
            max_samples_needed = (n_branches**2 - n_branches) // 2
            if max_samples_needed <= n_samples:
                print("    Sampling without replacement")
                with_replacement = False
            else:
                print("    Sampling with replacement")
                with_replacement = True

            checked_pairs = []
            n_samples_actual = 0
            for i in range(min(max_samples_needed, n_samples)):
                branch1, branch2 = None, None
                trial = 0

                while (
                    branch1 == branch2
                    or branch1 is None
                    or branch2 is None
                    or (branch1.ID, branch2.ID) in checked_pairs
                ):
                    if trial > 500:
                        break
                    branch1 = self.get_random_leaf(prob_weighted=with_replacement)
                    branch2 = self.get_random_leaf(prob_weighted=with_replacement)
                    trial += 1

                if (
                    branch1 == branch2
                    or branch1 is None
                    or branch2 is None
                    or (branch1.ID, branch2.ID) in checked_pairs
                ):
                    print(f"    breaking early after {n_samples_actual} samples")
                    break

                if not with_replacement:
                    checked_pairs += [(branch1.ID, branch2.ID), (branch2.ID, branch1.ID)]

                site_ind = branch1.tebd_engine.psi.L // 2
                site_inds = [site_ind, site_ind + 1]
                psi1 = branch1.tebd_engine.psi
                psi2 = branch2.tebd_engine.psi
                overlap = measure.trace_norm(partial_overlap(psi1, psi2, site_inds))

                if not with_replacement:
                    overlap *= psi1.norm * psi2.norm

                error += overlap
                n_samples_actual += 1

            if with_replacement:
                error /= n_samples_actual

            self.trace_distances["estimated_interference_error"] = error
            print(f"    estimated_interference_error = {error}")
            return error

    def set_all_unsynchronized(self):
        self.synchronized = False
        for child in self.children:
            child.set_all_unsynchronized()

    def determine_if_finished(self):
        if len(self.children) == 0:
            return self.finished
        else:
            self.finished = all([child.determine_if_finished() for child in self.children])
            return self.finished

    def determine_if_synchronized(self):
        if len(self.children) == 0:
            return self.synchronized
        else:
            self.synchronized = all([child.determine_if_synchronized() for child in self.children])
            return self.synchronized

    def get_unfinished_leaf_nodes_and_probs(self):
        if self.finished or self.synchronized:
            return []
        elif len(self.children) == 0:
            return [(self, np.abs(self.tebd_engine.psi.norm) ** 2)]
        else:
            nodes_probs_list = []
            for child in self.children:
                nodes_probs_list += child.get_unfinished_leaf_nodes_and_probs()
            return nodes_probs_list

    def get_next_branch_to_evolve(self):
        nodes_probs_list = self.get_unfinished_leaf_nodes_and_probs()
        if len(nodes_probs_list) == 0:
            return None
        else:
            nodes_probs_list.sort(key=lambda x: x[1])
            return nodes_probs_list[-1][0]

    def evolve_and_branch(self, stop_before_branching=False, t_evo=None, **kwargs):
        """The master function for evolving and branching the wavefunction. This should only be called on the root node."""
        if self.parent is not None:
            warnings.warn(
                f"WARNING: {self.ID}I am not a root node: I have a parent. The function evolve_and_branch() is only intended to be called on the root node. Alternatively, call evolve_and_branch_leaf() if you want to update leaf nodes individually. CONTINUING ANYWAY."
            )

        print(f"{self.ID}Starting root node time evolution and branching.")
        while not self.finished:
            self.estimate_inteference_error()

            self.determine_if_finished()
            if self.finished:
                print("EVERYTHING FINISHED")
                self.print_tree()
                self.save(final=True)
                self.plot()
                return

            self.determine_if_synchronized()
            if self.synchronized:
                print("EVERYTHING SYNCHRONIZED")
                self.print_tree()
                self.save()
                self.plot()
                self.set_all_unsynchronized()

            child_to_evolve = self.get_next_branch_to_evolve()
            if child_to_evolve is None:
                self.print_tree()
                self.save()
                self.plot()
                print("Couldn't find any more branches to evolve.")
                return
            child_to_evolve.evolve_and_branch_leaf(
                stop_before_branching=stop_before_branching, t_evo=t_evo, **kwargs
            )

    def evolve_and_branch_leaf(self, stop_before_branching=False, t_evo=None, **kwargs):
        """The leaf-node function for evolving and branching the wavefunction. This should only be called on a branch with no children (a leaf node)."""
        if len(self.children) != 0:
            warnings.warn(
                f"ERROR: {self.ID}I am not a leaf node: I have children. Please only call evolve_and_branch_leaf() on leaf nodes, or call evolve_and_branch() on the root node. RETURNING."
            )
            return

        print(f"{self.ID}Starting time evolution.")
        while (
            self.evolved_time < self.cfg.t_evo
            and len(self.children) == 0
            and not self.synchronized
            and not self.finished
        ):
            # Measure
            self.branch_values.add_measurements_tebd(
                self.tebd_engine, extra_measurements=self.trace_distances
            )

            # Evolve
            if np.random.randint(50) == 0:
                print(f"{self.name}:\n{self.get_root().print_tree(highlight_id=self.ID)}")
            self.tebd_engine.run_evolution(self.cfg.steps_per_output, self.dt)
            self.evolved_time = float(abs(self.tebd_engine.evolved_time))

            # Wait to synchronize with other branches if necessary
            if (
                self.evolved_time - self.time_of_last_synchronization
            ) >= self.cfg.synchronization_time:
                print(
                    f"{self.ID}Synchronized: evolved_time = {self.evolved_time}, time_of_last_synchronization = {self.time_of_last_synchronization}, synchronization_time = {self.cfg.synchronization_time}"
                )
                self.synchronized = True
                self.time_of_last_synchronization = self.evolved_time
                break

            # Branch
            if self.branch_function is not None and (
                self.cfg.max_branching_attempts is None
                or self.branching_attempts < self.cfg.max_branching_attempts
            ):
                # print(f'{self.ID}t_last_attempted_branching = {self.t_last_attempted_branching} ( / {self.cfg.min_time_between_branching_attempts / (self.tebd_engine.psi.L / 2)} ) | t_last_attempted_branching_sites = {self.t_last_attempted_branching_sites}')
                if (
                    self.evolved_time - self.t_last_attempted_branching
                    >= self.cfg.min_time_between_branching_attempts / (self.tebd_engine.psi.L / 2)
                ):
                    # print(f'{self.ID}Finding regions to branch')
                    # Find regions to branch
                    coarsegrain_from, coarsegrain_size = self.find_region_to_branch()

                    # If we've found regions to branch
                    if coarsegrain_from is not None:
                        assert coarsegrain_size is not None
                        if stop_before_branching:
                            print(
                                f"{self.ID}Stopping before the first branching: t = {self.evolved_time}"
                            )
                            return
                        # Branch
                        print(f"{self.ID}t = {self.evolved_time:.3f}")
                        print(
                            f"{self.ID}Branching between sites {coarsegrain_from} and {coarsegrain_from + coarsegrain_size - 1}"
                        )
                        self.t_last_attempted_branching = self.evolved_time
                        self.branching_attempts += 1
                        for i in range(coarsegrain_from, coarsegrain_from + coarsegrain_size):
                            self.t_last_attempted_branching_sites[
                                i % len(self.t_last_attempted_branching_sites)
                            ] = self.evolved_time
                        print(
                            f"{self.ID}t_last_attempted_branching_sites = {self.t_last_attempted_branching_sites}"
                        )
                        self.branch_and_sample(
                            coarsegrain_from=coarsegrain_from,
                            coarsegrain_size=coarsegrain_size,
                            **kwargs,
                        )

                        # Let the root node handle evolution of the children
                        if len(self.children) > 0:
                            print(f"{self.ID}Breaking now that I have children.")
                            break

        if len(self.children) == 0:
            if self.evolved_time >= self.cfg.t_evo:
                print(f"{self.ID}Finished.")
                self.finished = True
                # Measure
                self.branch_values.add_measurements_tebd(
                    self.tebd_engine, extra_measurements=self.trace_distances
                )
        # Return so that the the root node to decide which branch to evolve next
        return


def main(
    iterative_method: None
    | Literal[
        "bell_discard_classical",
        "bell_keep_classical",
        "vertical_svd_micro_bsvd",
        "pulling_through",
    ] = "vertical_svd_micro_bsvd",
    graddesc_method: None
    | Literal[
        "rho_LM_MR_trace_norm_discard_classical_identical_blocks",
        "rho_LM_MR_trace_norm_identical_blocks",
        "rho_LM_MR_trace_norm",
        "graddesc_global_reconstruction_non_interfering",
        "graddesc_global_reconstruction_split_non_interfering",
    ] = None,
    J=1.0,
    g=2.0,
    chi_max=180,
    n_sites=16,
    BC_MPS="finite",
    dt=0.005,
    N_steps_per_output=5,
    svd_min=1e-7,
    trunc_cut=1e-5,
    evo_method: Literal["TEBD", "MPO"] = "TEBD",
    branching=True,
    max_branches=8,
    chi_to_branch=50,
    max_trace_distance=1.0,
    max_overlap_error=1.0,
    t_evo=10.0,
    min_time_between_branching_attempts=0.0,
    max_branching_attempts=None,
    stop_before_branching=False,
    maxiter_heuristic=500,
    name="",
    outfolder=None,
    discard_branch_prob_factor=1e-2,
    synchronization_time=1.0,
    save_full_state=False,
    necessary_local_truncation_improvement_factor=1.1,
    necessary_global_truncation_improvement_factor=1.1,
    seed=None,
):
    print("Running main")
    print(f"Iterative method: {iterative_method}")
    print(f"Gradient descent method: {graddesc_method}")

    # Handle normal truncation evolution (no branching)
    if iterative_method is None or iterative_method == "None":
        max_branches = 1
        chi_to_branch = 999999999
        min_time_between_branching_attempts = 999999999
        max_branching_attempts = -1

    if seed is not None:
        np.random.seed(seed=seed)

    if branching in ["false", "False", "null", "none", "None"]:
        branching = False
    if stop_before_branching in ["false", "False", "null", "none", "None"]:
        stop_before_branching = False
    if max_branching_attempts in ["null", "none", "None"]:
        max_branching_attempts = None
    if outfolder in ["null", "none", "None"] or outfolder is None:
        outfolder = DEFAULT_OUT_FOLDER
    else:
        outfolder = Path(outfolder)

    if name == "":
        name = f"{iterative_method}_{graddesc_method}_L{n_sites}_n{max_branches}"

    outfolder.mkdir(exist_ok=True, parents=True)
    pickle_folder = outfolder / "pickles"
    pickle_folder.mkdir(exist_ok=True, parents=True)
    pickle_file = pickle_folder / (name + ".pkl")

    if evo_method == "TEBD":
        Model = TFIChain
    else:
        Model = TFIModel
    model = Model(
        {"L": n_sites, "J": J, "g": g, "bc_MPS": BC_MPS, "conserve": None, "sort_charge": False}
    )
    sites = model.lat.mps_sites()
    psi = tenpy.networks.MPS.from_product_state(
        sites, [[1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]] * n_sites, bc=BC_MPS
    )
    tebd_params = {
        "start_time": 0.0,
        "order": 2,
        "dt": dt,
        "N_steps": N_steps_per_output,
        "trunc_params": {
            "chi_max": chi_max,
            "svd_min": svd_min,
            "trunc_cut": trunc_cut,
        },
        "compression_method": "SVD",
    }

    if evo_method == "TEBD":
        tebd_engine = tebd.TEBDEngine(psi, model, tebd_params)
    elif evo_method == "MPO":
        tebd_engine = ExpMPOEvolution(psi, model, tebd_params)
    else:
        assert False, "unknown evo_method"

    cfg = BranchingMPSConfig(
        steps_per_output=N_steps_per_output,
        branching=branching,
        chi_to_branch=chi_to_branch,
        max_branches=max_branches,
        max_branching_attempts=max_branching_attempts,
        max_trace_distance=max_trace_distance,
        max_overlap_error=max_overlap_error,
        t_evo=t_evo,
        min_time_between_branching_attempts=min_time_between_branching_attempts,
        discard_branch_prob_factor=discard_branch_prob_factor,
        synchronization_time=synchronization_time,
        save_full_state=save_full_state,
        necessary_local_truncation_improvement_factor=necessary_local_truncation_improvement_factor,
        necessary_global_truncation_improvement_factor=necessary_global_truncation_improvement_factor,
    )

    branch_function = partial(
        branch,
        iterative_method=iterative_method,
        graddesc_method=graddesc_method,
        n_steps_graddesc=maxiter_heuristic,
    )

    print("\n\n\n\nBranching evolution:")
    branching_MPS = BranchingMPS(
        tebd_engine,
        cfg,
        branch_function=branch_function,
        pickle_file=pickle_file,
        outfolder=outfolder,
        name=name,
        info={
            "iterative_method": iterative_method,
            "graddesc_method": graddesc_method,
            "n_steps_graddesc": maxiter_heuristic,
            "evo_method": evo_method,
            "n_sites": n_sites,
            "J": J,
            "g": g,
            "chi_max": chi_max,
            "svd_min": svd_min,
            "trunc_cut": trunc_cut,
            "BC_MPS": BC_MPS,
        },
    )
    branching_MPS.evolve_and_branch(stop_before_branching=stop_before_branching)
    branching_MPS.branch_values.combine_measurements()
    branching_MPS.save()
    branching_MPS.plot()

    print("\n\nCOMPLETELY FINISHED.")

    return branching_MPS


def print_args_kwargs_then_run_main(*args, **kwargs):
    if len(args) > 0:
        print("args: ")
        for i in range(len(args)):
            print(f"    arg {i}:  {args[i]}  |  type = {type(args[i]).__name__}")
    if len(kwargs) > 0:
        print("kwargs: ")
        for k, v in kwargs.items():
            print(f"    {k}:  {v}  |  type = {type(v).__name__}")
    main(*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(print_args_kwargs_then_run_main)

    # for iterative_method in ['bell_discard_classical']:
    #     print_args_kwargs_then_run_main(iterative_method, "rho_LM_MR_trace_norm_discard_classical_identical_blocks", t_evo = 2.6, chi_max=30, chi_to_branch=30, max_branches=100, n_sites=30, BC_MPS='finite', maxiter_heuristic=500, min_time_between_branching_attempts=2, synchronization_time=5)

    # for iterative_method in 'bell_discard_classical', 'bell_keep_classical', 'vertical_svd_micro_bsvd', 'pulling_through':
    #     print_args_kwargs_then_run_main(iterative_method, None, t_evo = 2.6, chi_max=30, chi_to_branch=30, max_branches=100, n_sites=30, BC_MPS='finite', maxiter_heuristic=500, min_time_between_branching_attempts=2, synchronization_time=5)

    # for graddesc_method in 'rho_LM_MR_trace_norm_discard_classical_identical_blocks', 'rho_LM_MR_trace_norm_identical_blocks', 'rho_half_LM_MR_trace_norm', 'graddesc_global_reconstruction_non_interfering', 'graddesc_global_reconstruction_split_non_interfering':
    #     print_args_kwargs_then_run_main('pulling_through', graddesc_method, t_evo = 2.6, chi_max=30, chi_to_branch=30, max_branches=100, n_sites=30, BC_MPS='finite', maxiter_heuristic=500, min_time_between_branching_attempts=2, synchronization_time=5)

    # print_args_kwargs_then_run_main(t_evo = 3.3, chi_max=40, branch_function_name='bell', chi_to_branch=35, max_branches=2000, n_sites=2, BC_MPS='infinite', tolEntropy=1e-2, tolNegativity=-999.999, maxiter_heuristic=400, min_time_between_branching_attempts=0.2, synchronization_time=4)
    # print_args_kwargs_then_run_main(t_evo = 5, chi_max=50, branch_function_name='bell_keep_classical', chi_to_branch=30, max_branches=100, n_sites=30, BC_MPS='finite', tolEntropy=1.0, tolNegativity=-999.999, maxiter_heuristic=800, min_time_between_branching_attempts=0.6, synchronization_time=3)

    # print_args_kwargs_then_run_main(t_evo = 1.9, chi_max=40, branch_function_name='blockdiag_2svals_rho_half', chi_to_branch=3500, max_branches=1, n_sites=6, BC_MPS='infinite', tolEntropy=2e-2, tolNegativity=-999.999, maxiter_heuristic=400, min_time_between_branching_attempts=0.25, synchronization_time=10.0)
    # print_args_kwargs_then_run_main(t_evo = 4, chi_max=40, branch_function_name='bsvd', chi_to_branch=35, max_branches=10000, n_sites=20, BC_MPS='finite', max_trace_distance= 0.01, min_time_between_branching_attempts=0.2)

# %%
