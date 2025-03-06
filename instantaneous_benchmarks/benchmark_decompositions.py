# %%
"""A benchmark suite for simultaneous block diagonalization and simulateous block SVD algorithms
These algorithms take a set of matrices {A_1 ... A_N} and perform similarity transformations
to yield a set of block diagonal matrices {B_1 ... B_n} with the same finest block structure.

This test suite depends on in a directory.json file which points to the test input matrices.
The input matrices and block_diagonal_test_data/directory.json should be generated with
generate_test_inputs.py before running this script.

This script produces a benchmark_results.json file with the metrics of each of the
block-diagonalization methods on each of the test inputs. The block diagonalization methods are
those defined inside the get_blockdiag_methods() function below.

For analysis and plots of the benchmark results, see the plot_benchmarks.py script.
"""

import copy

# import importlib
import json

# from sklearn.utils import shuffle
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Complex
from opt_einops import einsum, rearrange

import wavefunction_branching.measure as measure
from wavefunction_branching.decompositions.decompositions import branch_from_theta
from wavefunction_branching.utils.tensors import make_json_serializable, make_square

MatrixStack = Complex[np.ndarray, "N_matrices dim_L dim_R"]
Matrix = Complex[np.ndarray, "dim_L dim_R"]
SBD_RETURN_TYPE = tuple[Matrix, MatrixStack, Matrix, list[int]]
PurificationMatrixStack = Complex[np.ndarray, "D_purification N_matrices dim_L dim_R"]


def get_spectra(matrix_stack, cutoff=1e-16):
    svals_l = np.linalg.svd(
        rearrange(matrix_stack, "p l r -> (p l) r"), full_matrices=False, compute_uv=False
    )
    svals_r = np.linalg.svd(
        rearrange(matrix_stack, "p l r -> l (p r)"), full_matrices=False, compute_uv=False
    )
    pl = int(np.sqrt(matrix_stack.shape[0]))
    svals_m = np.linalg.svd(
        rearrange(matrix_stack, "(pl pr) l r -> (pl l) (pr r)", pl=pl),
        full_matrices=False,
        compute_uv=False,
    )
    svals_l = svals_l[svals_l > cutoff]
    svals_m = svals_m[svals_m > cutoff]
    svals_r = svals_r[svals_r > cutoff]
    return svals_l, svals_m, svals_r


def positive_hermitian_matrix_power(matrix, power):
    evalsRight, evecsRight = np.linalg.eig(matrix)
    return einsum(
        evecsRight, np.abs(evalsRight) ** (power), np.linalg.inv(evecsRight), "L s, s, s R -> L R"
    )


def costFun_rho_half_LM_MR(purif_, orig):
    purif = purif_[:, : orig.shape[0], : orig.shape[1], : orig.shape[2]]
    left_env = einsum(
        orig,
        orig.conj(),
        " dPhys dVirt_L dVirt_R, dPhys_c dVirt_L_c dVirt_R"
        "->  dPhys dVirt_L          dPhys_c dVirt_L_c ",
    )

    right_env = einsum(
        orig,
        orig.conj(),
        "dPhys dVirt_L dVirt_R, dPhys_c dVirt_L dVirt_R_c "
        "-> dPhys         dVirt_R  dPhys_c         dVirt_R_c ",
    )

    left_env_half = positive_hermitian_matrix_power(
        rearrange(left_env, "p l pc lc -> (p l) (pc lc)"), 0.5
    )
    right_env_half = positive_hermitian_matrix_power(
        rearrange(right_env, "p r pc rc -> (p r) (pc rc)"), 0.5
    )

    rhoLeft = einsum(
        purif,
        purif.conj(),
        "nBlocks dPhys dVirt_L dVirt_R, nBlocks dPhys_c dVirt_L_c dVirt_R"
        "->         dPhys dVirt_L                  dPhys_c dVirt_L_c        ",
    )
    rhoRight = einsum(
        purif,
        purif.conj(),
        "nBlocks dPhys dVirt_L dVirt_R, nBlocks dPhys_c dVirt_L dVirt_R_c "
        "->         dPhys         dVirt_R          dPhys_c         dVirt_R_c ",
    )

    rhoLeft_half = positive_hermitian_matrix_power(
        rearrange(rhoLeft, "p l pc lc -> (p l) (pc lc)"), 0.5
    )
    rhoRight_half = positive_hermitian_matrix_power(
        rearrange(rhoRight, "p r pc rc -> (p r) (pc rc)"), 0.5
    )

    cost_left = np.sum(np.abs(rhoLeft_half - left_env_half) ** 2)
    cost_right = np.sum(np.abs(rhoRight_half - right_env_half) ** 2)

    cost = 0.5 * (cost_left + cost_right)
    return cost


def benchmark_blockdiag_method(
    method: Callable[[MatrixStack], PurificationMatrixStack], orig: MatrixStack
) -> dict:
    """Runs a decomposition method ("method") on a set of matrices ("orig") and returns the results"""
    t0 = time.time()
    purif: PurificationMatrixStack = method(orig)
    walltime = time.time() - t0

    assert purif.ndim == 4, (
        f"purif.ndim = {purif.ndim} - should be 4 (b, p, l, r) - purif.shape = {purif.shape}"
    )

    out = measure.LMR_trace_distances(orig, purif, measure_LR=False, measure_split=True)

    purif_pure = np.sum(purif, 0)
    out_pure = measure.LMR_trace_distances(
        orig, np.expand_dims(purif_pure, 0), measure_LR=False, measure_split=False
    )
    for key in out_pure:
        if "norm_error" not in key and "one_minus_overlap" not in key:
            out[key + "_pure"] = out_pure[key]

    out["walltime"] = walltime
    out["prob_orig"] = np.abs(np.conj(abs(orig.flatten() @ np.conj(orig.flatten()))))

    probs = np.zeros(purif.shape[0])
    for i in range(4):
        if i >= purif.shape[0]:
            out[f"prob_branch_{i}"] = 0.0
        else:
            probs[i] = np.abs(purif[i].flatten() @ np.conj(purif[i].flatten()))
            out[f"prob_branch_{i}"] = probs[i]
            out[f"norm_branch_{i}"] = np.sqrt(probs[i])
    out["total_prob"] = np.abs(np.sum(probs))
    out["average_prob"] = np.mean(probs)
    purif *= out["prob_orig"] / out["total_prob"]
    probs_prod = np.prod(probs / out["total_prob"])
    out["probs_prod"] = probs_prod
    out["probs_geometric_mean"] = probs_prod ** (1.0 / purif.shape[0])
    out["n_branches"] = purif.shape[0]

    out["costFun_LM_MR_rho_half_frobenius"] = costFun_rho_half_LM_MR(purif, orig)
    out["costFun_LM_MR_trace_distance"] = 0.5 * (
        out["trace_distance_LM"] + out["trace_distance_MR"]
    )

    out["costFun_split_trace_distance"] = (
        out["trace_distance_Lpl"] + out["trace_distance_M"] + out["trace_distance_Rpr"]
    ) / 3.0

    # Reconstruction error
    out["reconstruction_error_pure_frobenius"] = np.linalg.norm((purif_pure - orig).flatten())
    out["prob_branches_mixed"] = np.sum(
        np.abs(einsum(purif, np.conj(purif), "b p l r, b p l r -> b"))
    )
    out["prob_branches_pure"] = np.abs(einsum(purif_pure, np.conj(purif_pure), "p l r, p l r -> "))
    out["overlap_branches_pure_orig"] = np.abs(
        einsum(purif_pure, np.conj(orig), "p l r, p l r -> ")
    )
    out["global_reconstruction_error_trace_distance"] = np.sqrt(
        np.abs(
            1.0
            - np.abs(
                out["overlap_branches_pure_orig"] / (out["prob_branches_pure"] * out["prob_orig"])
            )
            ** 2
        )
    )

    purif_expanded = rearrange(
        purif, "b (p1 p2) l r -> b p1 p2 l r", p1=int(np.sqrt(purif.shape[1]))
    )
    out["overlap_branches"] = 0.0
    out["overlap_LR_branches"] = 0.0
    out["overlap_L_branches"] = 0.0
    out["overlap_R_branches"] = 0.0
    out["overlap_Lpl_branches"] = 0.0
    out["overlap_Rpr_branches"] = 0.0
    for i in range(4):
        for j in range(4):
            if i < j:
                if i >= purif.shape[0] or j >= purif.shape[0]:
                    out[f"overlap_branch_{i}_branch_{j}"] = 0.0
                    out[f"overlap_LR_branch_{i}_branch_{j}"] = 0.0
                    out[f"overlap_L_branch_{i}_branch_{j}"] = 0.0
                    out[f"overlap_R_branch_{i}_branch_{j}"] = 0.0
                    out[f"overlap_Lpl_branch_{i}_branch_{j}"] = 0.0
                    out[f"overlap_Rpr_branch_{i}_branch_{j}"] = 0.0
                else:
                    branch_i = copy.deepcopy(purif[i])
                    branch_j = copy.deepcopy(purif[j])
                    branch_i_expanded = copy.deepcopy(purif_expanded[i])
                    branch_j_expanded = copy.deepcopy(purif_expanded[j])
                    if out[f"norm_branch_{i}"] > 1e-14:
                        branch_i = branch_i / out[f"norm_branch_{i}"]
                        branch_i_expanded = branch_i_expanded / out[f"norm_branch_{i}"]
                    if out[f"norm_branch_{j}"] > 1e-14:
                        branch_j = branch_j / out[f"norm_branch_{j}"]
                        branch_j_expanded = branch_j_expanded / out[f"norm_branch_{j}"]
                    out[f"overlap_branch_{i}_branch_{j}"] = abs(
                        branch_i.flatten() @ np.conj(branch_j.flatten())
                    )
                    out[f"overlap_LR_branch_{i}_branch_{j}"] = measure.trace_norm(
                        einsum(branch_i, np.conj(branch_j), "p l r, pc l r  -> p   pc   ")
                    )
                    out[f"overlap_L_branch_{i}_branch_{j}"] = measure.trace_norm(
                        einsum(branch_i, np.conj(branch_j), "p l r, pc l rc -> p r pc rc")
                    )
                    out[f"overlap_R_branch_{i}_branch_{j}"] = measure.trace_norm(
                        einsum(branch_i, np.conj(branch_j), "p l r, pc lc r -> p l pc lc")
                    )

                    out[f"overlap_Lpl_branch_{i}_branch_{j}"] = measure.trace_norm(
                        einsum(
                            branch_i_expanded,
                            np.conj(branch_j_expanded),
                            "pl pr l r, pl prc l rc -> pr r prc rc",
                        )
                    )
                    out[f"overlap_Rpr_branch_{i}_branch_{j}"] = measure.trace_norm(
                        einsum(
                            branch_i_expanded,
                            np.conj(branch_j_expanded),
                            "pl pr l r, plc pr lc r -> pl l plc lc",
                        )
                    )

                    weight = out[f"norm_branch_{i}"] * out[f"norm_branch_{j}"]
                    out["overlap_branches"] += weight * out[f"overlap_branch_{i}_branch_{j}"]
                    out["overlap_LR_branches"] += weight * out[f"overlap_LR_branch_{i}_branch_{j}"]
                    out["overlap_L_branches"] += weight * out[f"overlap_L_branch_{i}_branch_{j}"]
                    out["overlap_R_branches"] += weight * out[f"overlap_R_branch_{i}_branch_{j}"]
                    out["overlap_Lpl_branches"] += (
                        weight * out[f"overlap_Lpl_branch_{i}_branch_{j}"]
                    )
                    out["overlap_Rpr_branches"] += (
                        weight * out[f"overlap_Rpr_branch_{i}_branch_{j}"]
                    )

    out["interference_error_trace_distance"] = (
        out["overlap_Lpl_branches"] + out["overlap_LR_branches"] + out["overlap_Rpr_branches"]
    )

    out["reconstruction_plus_interference_error_trace_distance"] = (
        out["global_reconstruction_error_trace_distance"] + out["interference_error_trace_distance"]
    )

    # Compute the truncation spectrum (the singular values of the schmidt decompositions) before and after branching
    svals_l_orig, svals_m_orig, svals_r_orig = get_spectra(orig)
    out["svals_entropy_l_orig"] = -np.sum(
        [svals_l_orig[i] * np.log(svals_l_orig[i]) for i in range(len(svals_l_orig))]
    )
    out["svals_entropy_m_orig"] = -np.sum(
        [svals_m_orig[i] * np.log(svals_m_orig[i]) for i in range(len(svals_m_orig))]
    )
    out["svals_entropy_r_orig"] = -np.sum(
        [svals_r_orig[i] * np.log(svals_r_orig[i]) for i in range(len(svals_r_orig))]
    )
    svals_l_branches = []
    svals_m_branches = []
    svals_r_branches = []
    svals_entropies_l = []
    svals_entropies_m = []
    svals_entropies_r = []
    for i in range(purif.shape[0]):
        svals_l, svals_m, svals_r = get_spectra(purif[i])
        svals_l_branches.append(svals_l)
        svals_m_branches.append(svals_m)
        svals_r_branches.append(svals_r)
        svals_entropies_l.append(
            -np.sum([svals_l[i] * np.log(svals_l[i]) for i in range(len(svals_l))])
        )
        svals_entropies_m.append(
            -np.sum([svals_m[i] * np.log(svals_m[i]) for i in range(len(svals_m))])
        )
        svals_entropies_r.append(
            -np.sum([svals_r[i] * np.log(svals_r[i]) for i in range(len(svals_r))])
        )
    out["svals_entropy_l_sum"] = np.sum(svals_entropies_l)
    out["svals_entropy_m_sum"] = np.sum(svals_entropies_m)
    out["svals_entropy_r_sum"] = np.sum(svals_entropies_r)
    out["svals_entropy_l_sum_on_orig"] = np.sum(svals_entropies_l) / out["svals_entropy_l_orig"]
    out["svals_entropy_m_sum_on_orig"] = np.sum(svals_entropies_m) / out["svals_entropy_m_orig"]
    out["svals_entropy_r_sum_on_orig"] = np.sum(svals_entropies_r) / out["svals_entropy_r_orig"]

    for key in [
        "walltime",
        "costFun_LM_MR_trace_distance",
        "costFun_split_trace_distance",
        "global_reconstruction_error_trace_distance",
        "interference_error_trace_distance",
        "reconstruction_plus_interference_error_trace_distance",
        "reconstruction_error_pure_frobenius",
    ]:
        print(f"{key}: {out[key]}")

    # Plot the truncation spectra
    # plt.figure(dpi=120)
    # palette = sns.color_palette()
    # plt.plot(svals_l_orig, c=palette[0], linestyle=':',  label='original left', alpha=0.5)
    # plt.plot(svals_m_orig, c=palette[0],                 label='original middle', alpha=0.5)
    # plt.plot(svals_r_orig, c=palette[0], linestyle='--', label='original right', alpha=0.5)
    # for i in range(purif.shape[0]):
    #     plt.plot(svals_l_branches[i], c=palette[i+1], linestyle=':',  label=f'branch {i} left', alpha=0.5)
    #     plt.plot(svals_m_branches[i], c=palette[i+1],                 label=f'branch {i} middle', alpha=0.5)
    #     plt.plot(svals_r_branches[i], c=palette[i+1], linestyle='--', label=f'branch {i} right', alpha=0.5)
    # plt.ylabel('Singular values from Schmidt decomposition')
    # plt.xlabel('Singular value number')
    # plt.yscale('log')
    # plt.legend()
    # plt.show()

    # rho_orig_expanded  = einsum(orig_expanded,  np.conj(orig_expanded),   ' p1 p2 l r,   p1c p2c l r -> p1 p2 p1c p2c')
    # rho_purif_expanded = einsum(purif_expanded, np.conj(purif_expanded), 'b p1 p2 l r, b p1c p2c l r -> p1 p2 p1c p2c')
    # measurement_props_orig = measure.calculate_properties_2site_density_matrix(rho_orig_expanded)
    # measurement_props_purif = measure.calculate_properties_2site_density_matrix(rho_purif_expanded)

    # eps = 1e-14
    # measurement_errors = {}
    # for key in measurement_props_orig:
    #     measurement_errors[key] = np.abs(measurement_props_orig[key] - measurement_props_purif[key])/(np.abs(measurement_props_orig[key])+eps)

    # for key in measurement_props_orig:
    #     print(f'    {key}: orig={measurement_props_orig[key]} | purif={measurement_props_purif[key]} | error={measurement_errors[key]}')

    # out.update({str(k) + '_orig' : np.real(v) for k,v in measurement_props_orig.items()})
    # out.update({str(k) + '_purif' : np.real(v) for k,v in measurement_props_purif.items()})
    # out.update({str(k) + '_error' : np.real(v) for k,v in measurement_errors.items()})

    return out


def get_blockdiag_methods() -> dict[str, Callable[[np.ndarray], PurificationMatrixStack]]:
    """
    The block-diagonalization methods to try.
    Each function defined within this function is added to a methods dict.
    Each function must take in an "As" np array of shape (N_matrices, dim_L, dim_R),
    and return a tuple of U, Bs, Vh, block_sizes, where As = U @ Bs @ Vh.
    """

    def truncation(As: MatrixStack, dim_factor=0.5) -> PurificationMatrixStack:
        LM, svals_LM_R, R = np.linalg.svd(rearrange(As, "p l r -> (p l) r"))
        svals_LM_R = svals_LM_R[: int(As.shape[-1] * dim_factor)]
        R = R[: len(svals_LM_R), :]
        L, svals_L_MR, MR = np.linalg.svd(rearrange(As, "p l r -> l (p r)"))
        svals_L_MR = svals_L_MR[: int(As.shape[-2] * dim_factor)]
        L = L[:, : len(svals_L_MR)]

        As_trunc = einsum(np.conj(L), As, np.conj(R), "L l, p L R, r R -> p l r")

        theta = einsum(L, As_trunc, R, "L l, p l r, r R -> p L R")
        theta_with_zero_branch = np.zeros([2] + list(theta.shape), dtype=theta.dtype)
        theta_with_zero_branch[0] = theta
        return theta_with_zero_branch

    def bell_discard_classical(As: MatrixStack) -> PurificationMatrixStack:
        return branch_from_theta(As, "bell_discard_classical", None)[0]

    def bell_keep_classical(As: MatrixStack) -> PurificationMatrixStack:
        return branch_from_theta(As, "bell_keep_classical", None)[0]

    def vertical_svd_micro_bsvd(As: MatrixStack) -> PurificationMatrixStack:
        return branch_from_theta(As, "vertical_svd_micro_bsvd", None)[0]

    def pulling_through(As: MatrixStack) -> PurificationMatrixStack:
        return branch_from_theta(As, "pulling_through", None)[0]

    def bell_discard_classical__rho_LM_MR_trace_norm_discard_classical_identical_blocks(
        As: MatrixStack,
    ) -> PurificationMatrixStack:
        return branch_from_theta(
            As, "bell_discard_classical", "rho_LM_MR_trace_norm_discard_classical_identical_blocks"
        )[0]

    def bell_keep_classical__rho_LM_MR_trace_norm_identical_blocks(
        As: MatrixStack,
    ) -> PurificationMatrixStack:
        return branch_from_theta(
            As, "bell_keep_classical", "rho_LM_MR_trace_norm_identical_blocks"
        )[0]

    def bell_keep_classical__rho_LM_MR_trace_norm(As: MatrixStack) -> PurificationMatrixStack:
        return branch_from_theta(As, "bell_keep_classical", "rho_LM_MR_trace_norm")[0]

    def vertial_svd_micro_bsvd__rho_LM_MR_trace_norm(As: MatrixStack) -> PurificationMatrixStack:
        return branch_from_theta(As, "vertical_svd_micro_bsvd", "rho_LM_MR_trace_norm")[0]

    def pulling_through__rho_LM_MR_trace_norm(As: MatrixStack) -> PurificationMatrixStack:
        return branch_from_theta(As, "pulling_through", "rho_LM_MR_trace_norm")[0]

    # def bell_keep_classical__rho_half_LM_MR_trace_norm(As: MatrixStack) -> PurificationMatrixStack:
    #     return branch_from_theta(As, "bell_keep_classical", "rho_half_LM_MR_trace_norm")[0]

    # def vertial_svd_micro_bsvd__rho_half_LM_MR_trace_norm(As: MatrixStack) -> PurificationMatrixStack:
    #     return branch_from_theta(As, "vertical_svd_micro_bsvd", "rho_half_LM_MR_trace_norm")[0]

    # def pulling_through__rho_half_LM_MR_trace_norm(As: MatrixStack) -> PurificationMatrixStack:
    #     return branch_from_theta(As, "pulling_through", "rho_half_LM_MR_trace_norm")[0]

    def bell_keep_classical__graddesc_global_reconstruction_non_interfering(
        As: MatrixStack,
    ) -> PurificationMatrixStack:
        return branch_from_theta(
            As, "bell_keep_classical", "graddesc_global_reconstruction_non_interfering"
        )[0]

    def vertial_svd_micro_bsvd__graddesc_global_reconstruction_non_interfering(
        As: MatrixStack,
    ) -> PurificationMatrixStack:
        return branch_from_theta(
            As, "vertical_svd_micro_bsvd", "graddesc_global_reconstruction_non_interfering"
        )[0]

    def pulling_through__graddesc_global_reconstruction_non_interfering(
        As: MatrixStack,
    ) -> PurificationMatrixStack:
        return branch_from_theta(
            As, "pulling_through", "graddesc_global_reconstruction_non_interfering"
        )[0]

    def bell_keep_classical__graddesc_global_reconstruction_split_non_interfering(
        As: MatrixStack,
    ) -> PurificationMatrixStack:
        return branch_from_theta(
            As, "bell_keep_classical", "graddesc_global_reconstruction_split_non_interfering"
        )[0]

    def vertial_svd_micro_bsvd__graddesc_global_reconstruction_split_non_interfering(
        As: MatrixStack,
    ) -> PurificationMatrixStack:
        return branch_from_theta(
            As, "vertical_svd_micro_bsvd", "graddesc_global_reconstruction_split_non_interfering"
        )[0]

    def pulling_through__graddesc_global_reconstruction_split_non_interfering(
        As: MatrixStack,
    ) -> PurificationMatrixStack:
        return branch_from_theta(
            As, "pulling_through", "graddesc_global_reconstruction_split_non_interfering"
        )[0]

    def do_nothing(As: MatrixStack) -> PurificationMatrixStack:
        return np.expand_dims(As, 0)

    # Return a dict of functions defined within this function
    methods = {}
    for key, value in locals().items():
        if callable(value) and value.__module__ == __name__:
            methods[key] = value
    return methods


def modify_path_str(current_path, path_str):
    return current_path / Path(
        path_str
    )  # Path(path_str.split('wavefunction_branching\\')[1].replace('\\', '/'))


if __name__ == "__main__":
    # Block diagonalization methods to try
    methods = get_blockdiag_methods()

    print(f"Methods to test: {list(methods.keys())}")
    max_method_name_len = max([len(x) for x in methods.keys()])

    # Dict to store the results
    results = defaultdict(list)
    spectra = defaultdict(list)

    # Get the directory of sets of "As" matrices to try to simultaneously block-diagonalize
    current_path = Path(__file__).parent.absolute()
    outfolder = current_path / "benchmark_results"
    outfolder.mkdir(exist_ok=True)
    outfile = outfolder / f"benchmark_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    directory = json.load(
        open(current_path / "block_diagonal_test_data/directory.json", mode="rb")
    )  # block_diagonal_test_data-everything_2024-08-21
    print([(x, len(directory[x])) for x in directory])
    directory = pd.DataFrame(directory)
    # directory = shuffle(directory)
    # directory.sort_values('second-dominant TM eigenvalue magnitude', inplace=True, ascending=False)
    # directory.sort_values('dim_L', inplace=True)
    # directory = directory[directory['kind'] == 'ising_evo']
    print(len(directory))
    directory = directory[directory["t"] >= 2.5]
    directory = directory[directory["dim_L"] > 15]
    # directory = directory[directory['second-dominant TM eigenvalue magnitude'] >= 0.5]
    print(len(directory))
    print(directory)

    # Filter the directory for what we're testing
    # directory = directory[directory['scramble_kind'].isin(['UV','null'])]
    # directory = directory[directory['kind'] == 'ising_evo']
    # directory = directory.sort_values('t')
    # Shuffle the directory's rows
    # directory = directory.iloc[np.random.permutation(len(directory))]
    # Loop over the directory of sets of "As" matrices to try to simultaneously block-diagonalize

    #     df = directory[[('g0=inf' in x) for x in directory['save_str']]]
    #     plt.rcParams['figure.figsize'] = [10,10]
    #     sns.lineplot(df, x='average bond dimension', y='second-dominant TM eigenvalue magnitude', hue='kind', alpha=0.7)#, style='kind')
    #     plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    #     plt.show()
    #     sns.lineplot(df, x='t', y='second-dominant TM eigenvalue magnitude', hue='kind', alpha=0.7)#, style='kind')
    #     plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    #     plt.show()

    # %%
    i = 0
    spectra_L = []
    spectra_R = []
    for r in directory.index:
        i += 1
        row = directory.loc[r]
        print(f"({i}/{len(directory)}) Testing row {r}: {row.save_str}")

        # Validate the path to the As matrices
        As_path = row.As_file
        assert isinstance(As_path, str) and As_path != "null"
        As_path = modify_path_str(current_path, As_path)
        assert As_path.exists(), f"File {As_path} does not exist."

        # Load the As matrices to simultaneously block diagonalize
        As = np.load(As_path)
        # Do sanity checks on the As matrices
        assert len(As.shape) == 3, f"As must be a three-index tensor in {row.save_str}"
        assert As.shape[0] == row.N_matrices, f"As.shape[0] != row.N_matrices in {row.save_str}"
        if row.dim_L is not None and row.dim_L != "null":
            assert As.shape[1] == row.dim_L, f"As.shape[1] != row.dim_L in {row.save_str}"
        if row.dim_R is not None and row.dim_R != "null":
            assert As.shape[2] == row.dim_R, f"As.shape[2] != row.dim_R in {row.save_str}"

        # Load the singular values if we have them
        svals_L_path = row.svals_L_file
        if svals_L_path == "null" or not isinstance(svals_L_path, str):
            svals_L = np.ones(As.shape[-2])
        else:
            svals_L = np.load(modify_path_str(current_path, svals_L_path))
            print(f"svals_L = {svals_L}")

        svals_R_path = row.svals_R_file
        if svals_R_path == "null" or not isinstance(svals_R_path, str):
            svals_R = np.ones(As.shape[-1])
        else:
            svals_R = np.load(modify_path_str(current_path, svals_R_path))
            print(f"svals_R = {svals_R}")

        # Correct for the gauge so that svals_L @ As @ svals_R is in central gauge
        form = row.form
        if form != "null":
            form_L, form_R = form
            svals_L = svals_L ** (1.0 - form_L)
            svals_R = svals_R ** (1.0 - form_R)

        # Load the ground truth Bs matrices to compare against if we have them
        Bs_orig = None
        Bs_path = row.Bs_file
        if isinstance(Bs_path, str) and Bs_path != "null":
            Bs_path = modify_path_str(current_path, Bs_path)
            if Bs_path.exists():
                Bs_orig = np.load(Bs_path)
                assert Bs_orig.shape == As.shape, (
                    f"Bs must be the same shape as As in {row.save_str}"
                )

        As_square = make_square(As, 2)

        # Do the block diagonalization for each method
        for method_name, method in methods.items():
            try:
                # Test the method
                print(
                    f"    \n\n{method_name}:" + " " * (max_method_name_len - len(method_name)),
                    end="",
                )
                np.random.seed(0)
                out = benchmark_blockdiag_method(method, As_square)
                # Add results to the results dict
                results["method_name"].append(method_name)
                method_name_split = method_name.split("__")
                results["iterative_method"].append(method_name_split[0])
                if len(method_name_split) > 1:
                    results["graddesc_method"].append(method_name_split[1])
                else:
                    results["graddesc_method"].append("None")
                for key, value in out.items():
                    results[key].append(value)
                for key, value in row.items():
                    results[key].append(value)
                # Print where we're at
                print(f" {out}")
            except Exception as e:
                print(f"\n\nWARNING: {method_name} failed: \n{e}\n")
                traceback.print_exc()
        # output to results.json
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4, default=make_json_serializable)

        # Get the spectrum for each test case
        spectrum_L = np.linalg.svd(rearrange(As, "p l r -> l (p r)"), compute_uv=False)
        spectrum_R = np.linalg.svd(rearrange(As, "p l r -> (p l) r"), compute_uv=False)
        print(f"spectrum_L = {spectrum_L}")
        print(f"spectrum_R = {spectrum_R}")
        spectra["spectrum_L"].append(spectrum_L)
        spectra["spectrum_R"].append(spectrum_R)
        for key, value in row.items():
            spectra[key].append(value)

    # # Plot the spectrum of each test case
    # plt.rcParams['figure.figsize'] = [10, 9]
    # shown_artificial = False
    # shown_natural1 = False
    # key = 'spectrum_L'
    # for i in range(len(spectra[key])):
    #     c = 'purple'
    #     label = None
    #     if spectra['form_L'][i] == 'null':
    #         c = 'red'
    #         if not shown_artificial:
    #             label = 'artificial'
    #             shown_artificial = True
    #     elif spectra['form_L'][i] == 1.0:
    #         c = 'blue'
    #         if not shown_natural1:
    #             label = 'natural (λ^1.0 - central gauge)'
    #             shown_natural1 = True
    #     plt.plot(spectra[key][i], c=c, alpha=0.25, label=label)
    # plt.ylabel('Singular value')
    # plt.xlabel('Singular value index')
    # plt.legend()
    # plt.yscale('log')
    # plt.show()


# %%
