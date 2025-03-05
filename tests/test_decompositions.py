# %%
import numpy as np
import pytest
from opt_einops import einsum
from scipy.stats import unitary_group  # Random unitaries

from instantaneous_benchmarks.benchmark_decompositions import benchmark_blockdiag_method
from instantaneous_benchmarks.generate_test_inputs import (
    block_diagonal_matrices_exp_decaying_spectrum,
)
from wavefunction_branching.decompositions.decompositions import branch_from_theta

ITERATIVE_METHODS = (
    "bell_discard_classical",
    "bell_keep_classical",
    "vertical_svd_micro_bsvd",
    "pulling_through",
)

GRADDESC_METHODS = (
    "rho_LM_MR_trace_norm_discard_classical_identical_blocks",
    "rho_LM_MR_trace_norm_identical_blocks",
    "rho_LM_MR_trace_norm",
    "graddesc_global_reconstruction_non_interfering",
    "graddesc_global_reconstruction_split_non_interfering",
)

METHODS = {
    "bell_discard_classical": (None,)
    + tuple(x for x in GRADDESC_METHODS if "discard_classical" in x),
    "bell_keep_classical": (None,)
    + tuple(x for x in GRADDESC_METHODS if "discard_classical" not in x),
    "vertical_svd_micro_bsvd": (None,)
    + tuple(x for x in GRADDESC_METHODS if "identical_blocks" not in x),
    "pulling_through": (None,) + tuple(x for x in GRADDESC_METHODS if "identical_blocks" not in x),
}

METHODS_TUPLES = []
for k, v in METHODS.items():
    METHODS_TUPLES.extend([(k, v) for v in v])

IDENTICAL_BLOCKS_ITERATIVE_METHODS = (
    "bell_discard_classical",
    "bell_keep_classical",
)
IDENTICAL_BLOCKS_GRADDESC_METHODS = (
    None,
    "rho_LM_MR_trace_norm_discard_classical_identical_blocks",
    "rho_LM_MR_trace_norm_identical_blocks",
)


def is_identical_blocks_method(iterative_method: str, graddesc_method: str | None) -> bool:
    return graddesc_method in IDENTICAL_BLOCKS_GRADDESC_METHODS


NON_GLOBAL_RECONSTRUCTION_METHODS = (
    "bell_discard_classical",
    "rho_LM_MR_trace_norm_discard_classical_identical_blocks",
    "rho_LM_MR_trace_norm_identical_blocks",
    "rho_LM_MR_trace_norm",
)


def construct_methods():
    for iterative_method in ITERATIVE_METHODS:
        for graddesc_method in GRADDESC_METHODS:
            yield (iterative_method, graddesc_method)


def generate_test_tensor(
    N=4,
    branch_chi=5,
    noise_introduced=1e-5,
    identical_blocks=False,
):
    if identical_blocks:
        print("Identical blocks")
    theta_orig = block_diagonal_matrices_exp_decaying_spectrum(
        block_sizes=[branch_chi, branch_chi], N=N, noise_introduced=noise_introduced, decayrate=3.5
    )  # shape: N, branch_chi, branch_chi
    if identical_blocks:
        theta_orig[:, branch_chi:, branch_chi:] = theta_orig[:, :branch_chi, :branch_chi]

    theta_orig_norm = theta_orig.flatten() @ np.conj(theta_orig.flatten())
    theta_orig /= np.sqrt(theta_orig_norm)

    branch_a = np.zeros_like(theta_orig)
    branch_a[:, :branch_chi, :branch_chi] = theta_orig[:, :branch_chi, :branch_chi]
    branch_b = np.zeros_like(theta_orig)
    branch_b[:, branch_chi:, branch_chi:] = theta_orig[:, branch_chi:, branch_chi:]

    theta_orig_purified = np.zeros((2,) + theta_orig.shape, dtype=theta_orig.dtype)
    theta_orig_purified[0] = branch_a
    theta_orig_purified[1] = branch_b

    U = unitary_group.rvs(theta_orig.shape[1])
    Vh = unitary_group.rvs(theta_orig.shape[2])
    assert np.allclose(U @ U.conj().T, np.eye(U.shape[1]))
    assert np.allclose(Vh @ Vh.conj().T, np.eye(Vh.shape[1]))

    theta_scrambled = einsum(U, theta_orig, Vh, "L l, p l r, r R -> p L R")
    theta_scrambled_purified = einsum(U, theta_orig_purified, Vh, "L l, b p l r, r R -> b p L R")
    return theta_orig, theta_orig_purified, theta_scrambled, theta_scrambled_purified


def check_results(
    results: dict, iterative_method: str, graddesc_method: str | None, tolerance: float
) -> str | None:
    methods_str = f"iterative_method = {iterative_method}, graddesc_method = {graddesc_method}"
    failures = []

    # Check norms
    if abs(results["total_prob"] - 1.0) >= tolerance:
        failures.append(
            f"total_prob = {results['total_prob']}, expected close to 1.0 within {tolerance}"
        )

    if "discard_classical" not in iterative_method:
        if results["probs_geometric_mean"] <= 1e-2:
            failures.append(
                f"probs_geometric_mean = {results['probs_geometric_mean']}, expected > 1e-2"
            )
        if results["n_branches"] != 2:
            failures.append(f"n_branches = {results['n_branches']}, expected 2")
    else:
        if results["n_branches"] != 4:
            failures.append(f"n_branches = {results['n_branches']}, expected 4")

    # Check local operators
    if results["costFun_LM_MR_trace_distance"] >= tolerance:
        failures.append(
            f"costFun_LM_MR_trace_distance = {results['costFun_LM_MR_trace_distance']}, expected < {tolerance}"
        )

    # Check global reconstruction
    if iterative_method not in NON_GLOBAL_RECONSTRUCTION_METHODS:
        if results["reconstruction_error_pure_frobenius"] >= tolerance:
            failures.append(
                f"reconstruction_error_pure_frobenius = {results['reconstruction_error_pure_frobenius']}, expected < {tolerance}"
            )
        if results["global_reconstruction_error_trace_distance"] >= tolerance:
            failures.append(
                f"global_reconstruction_error_trace_distance = {results['global_reconstruction_error_trace_distance']}, expected < {tolerance}"
            )
        if results["one_minus_overlap_pure"] >= tolerance:
            failures.append(
                f"one_minus_overlap_pure = {results['one_minus_overlap_pure']}, expected < {tolerance}"
            )

    # Check interference
    if results["overlap_branches"] >= tolerance:
        failures.append(f"overlap_branches = {results['overlap_branches']}, expected < {tolerance}")
    if results["overlap_LR_branches"] >= tolerance:
        failures.append(
            f"overlap_LR_branches = {results['overlap_LR_branches']}, expected < {tolerance}"
        )

    if "discard_classical" not in iterative_method:
        if results["interference_error_trace_distance"] >= tolerance:
            failures.append(
                f"interference_error_trace_distance = {results['interference_error_trace_distance']}, expected < {tolerance}"
            )
        if results["overlap_L_branches"] >= tolerance:
            failures.append(
                f"overlap_L_branches = {results['overlap_L_branches']}, expected < {tolerance}"
            )
        if results["overlap_R_branches"] >= tolerance:
            failures.append(
                f"overlap_R_branches = {results['overlap_R_branches']}, expected < {tolerance}"
            )
        if results["overlap_Lpl_branches"] >= tolerance:
            failures.append(
                f"overlap_Lpl_branches = {results['overlap_Lpl_branches']}, expected < {tolerance}"
            )
        if results["overlap_Rpr_branches"] >= tolerance:
            failures.append(
                f"overlap_Rpr_branches = {results['overlap_Rpr_branches']}, expected < {tolerance}"
            )

    if failures:
        return f"{methods_str}:\n" + "\n".join(failures)
    return None


@pytest.mark.parametrize("iterative_method, graddesc_method", METHODS_TUPLES)
def test_method(
    iterative_method: str,
    graddesc_method: str | None,
    tolerance: float = 0.005,
    N: int = 4,
    branch_chi: int = 5,
    noise_introduced: float = 1e-8,
    n_trials: int = 5,
):
    print(f"\n\nTesting iterative_method = {iterative_method}, graddesc_method = {graddesc_method}")
    # Run three trials and collect results
    results = []
    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials}")
        theta_orig, theta_orig_purified, theta_scrambled, theta_scrambled_purified = (
            generate_test_tensor(
                N=N,
                branch_chi=branch_chi,
                noise_introduced=noise_introduced,
                identical_blocks=is_identical_blocks_method(iterative_method, graddesc_method),
            )
        )

        def branching_fn(theta_scrambled):
            theta_purified, info_ = branch_from_theta(
                theta_scrambled,
                iterative_method,  # type: ignore
                graddesc_method,  # type: ignore
                n_steps_iterative=70 * (trial + 1),
                n_steps_graddesc=70 * (trial + 1),
            )
            # Normalize the purified state
            norm_pure = einsum(theta_purified, np.conj(theta_purified), "b p l r, bc p l r -> ")
            theta_purified /= np.sqrt(norm_pure)
            # for key, value in info_.items():
            #     info[key] = value
            return theta_purified

        results.append(benchmark_blockdiag_method(branching_fn, theta_scrambled))

        # Check that there was at least one successful trial
        failure = check_results(results[-1], iterative_method, graddesc_method, tolerance)
        if failure is None:
            break
        else:
            print(f"FAILURE in trial {trial + 1}/{n_trials}: \n{failure}")

    assert failure is None, failure


def check_decompositions(N=4, branch_chi=5, noise_introduced=1e-8, n_trials=5):
    tolerance = 0.005  # (noise_introduced + 1e-4) * 50
    for iterative_method in ITERATIVE_METHODS:
        for graddesc_method in METHODS[iterative_method]:
            test_method(
                iterative_method,
                graddesc_method,
                tolerance=tolerance,
                N=N,
                branch_chi=branch_chi,
                noise_introduced=noise_introduced,
                n_trials=n_trials,
            )
    print("\n\nDone.")


# %%
