"""The final suite of iterative and gradient descent methods included in the paper."""

import time
import warnings
from typing import Literal

import numpy as np
from jaxtyping import Complex
from opt_einops import einsum, rearrange, repeat
from tenpy.networks.mps import MPS

import wavefunction_branching.decompositions.bell_different_blocks as bell_different_blocks
import wavefunction_branching.decompositions.bell_identical_blocks as bell_identical_blocks

# import wavefunction_branching.decompositions.bmm_2svals_rho_half as bmm_2svals_rho_half
import wavefunction_branching.decompositions.graddesc_global as graddesc_global  # optimize(tensors_orig, tensors_a, tensors_b) -> tensors_a, tensors_b, dict
import wavefunction_branching.decompositions.pulling_through as pulling_through
import wavefunction_branching.decompositions.vertical_svd as vertical_svd
import wavefunction_branching.utils.tensors as utils
from wavefunction_branching.type_aliases import (
    BlockDiagTensor,
    LeftSplittingTensor,
    MatrixStack,
    PurificationMatrixStack,
    RightSplittingTensor,
)


def LSR_to_purification(
    L: LeftSplittingTensor, S: BlockDiagTensor, R: RightSplittingTensor, keep_classical: bool = True
):
    if len(S.shape) > 3:
        tensor = einsum(
            L,
            S,
            R,
            "xFast dVirt_L dSlow_L, dPhys xFast dSlow_L dSlow_R, xFast dSlow_R dVirt_R"
            "-> xFast dPhys dVirt_L dVirt_R",
        )
    elif keep_classical:
        tensor = einsum(
            L,
            S,
            R,
            "xFast dVirt_L dSlow_L, dPhys dSlow_L dSlow_R, xFast dSlow_R dVirt_R"
            "-> xFast dPhys dVirt_L dVirt_R",
        )
    else:
        tensor = einsum(
            L,
            S,
            R,
            "xFast_L dVirt_L dSlow_L, dPhys dSlow_L dSlow_R, xFast_R dSlow_R dVirt_R"
            "-> xFast_L xFast_R dPhys dVirt_L dVirt_R",
        )

        tensor = rearrange(
            tensor,
            " xFast_L xFast_R dPhys dVirt_L dVirt_R-> (xFast_L xFast_R) dPhys dVirt_L dVirt_R",
        )
    return tensor


def make_S_square(
    L: LeftSplittingTensor, S: BlockDiagTensor, R: RightSplittingTensor
) -> tuple[LeftSplittingTensor, BlockDiagTensor, RightSplittingTensor]:
    dPhys, nBranches, dSlow_L, dSlow_R = S.shape
    dVirt_L, dVirt_R = L.shape[1], R.shape[2]

    if dSlow_L == dSlow_R:
        return L, S, R
    if dSlow_L > dSlow_R:
        S_square = np.zeros([dPhys, nBranches, dSlow_L, dSlow_L], dtype=S.dtype)
        S_square[:, :, :dSlow_L, :dSlow_R] = S
        R_square = np.zeros([nBranches, dSlow_L, dVirt_R], dtype=R.dtype)
        R_square[:, :dSlow_R] = R
        return L, S_square, R_square
    assert dSlow_R > dSlow_L
    S_square = np.zeros([dPhys, nBranches, dSlow_R, dSlow_R], dtype=S.dtype)
    S_square[:, :, :dSlow_L, :dSlow_R] = S
    L_square = np.zeros([nBranches, dVirt_L, dSlow_R], dtype=L.dtype)
    L_square[:, :, :dSlow_L] = L
    return L_square, S_square, R


############################################################################################################
# ITERATIVE INITIALIZATION METHODS
############################################################################################################


def expand_S(
    S: Complex[np.ndarray, "dPhys dim_l dim_r"],
) -> Complex[np.ndarray, "dPhys 2 dim_l dim_r"]:
    S_expanded = repeat(S, "dPhys dim_l dim_r -> dPhys b dim_l dim_r", b=2)
    return S_expanded


def bell_discard_classical(
    As: MatrixStack, n_steps=500
) -> tuple[LeftSplittingTensor, BlockDiagTensor, RightSplittingTensor, dict]:
    tensor = utils.make_square(As, 2)
    L, S, R, info = bell_identical_blocks.combined_optimization(
        tensor,
        n_attempts_iterative=1,
        n_iterations_per_attempt=n_steps,
        early_stopping=False,
        maxiter_heuristic=0,
        keep_classical_correlations=False,
    )
    assert len(S.shape) == 3  # dPhys, dSlow, dSlow
    S_expanded = expand_S(S)
    return *make_S_square(L, S_expanded, R), info


def bell_keep_classical(
    As: MatrixStack, n_steps=500
) -> tuple[LeftSplittingTensor, BlockDiagTensor, RightSplittingTensor, dict]:
    tensor = utils.make_square(As, 2)
    L, S, R, info = bell_identical_blocks.combined_optimization(
        tensor,
        n_attempts_iterative=1,
        n_iterations_per_attempt=n_steps,
        early_stopping=False,
        maxiter_heuristic=0,
        keep_classical_correlations=True,
    )
    assert len(S.shape) == 3  # dPhys, dSlow, dSlow
    S_expanded = expand_S(S)
    return *make_S_square(L, S_expanded, R), info


def bell_original_threshold_discard_classical(
    As: MatrixStack, n_steps=500
) -> tuple[LeftSplittingTensor, BlockDiagTensor, RightSplittingTensor, dict]:
    tensor = utils.make_square(As, 2)
    L, S, R, info = bell_identical_blocks.combined_optimization(
        tensor,
        n_attempts_iterative=1,
        n_iterations_per_attempt=n_steps,
        early_stopping=False,
        maxiter_heuristic=0,
        keep_classical_correlations=False,
        tolEntropy=1e-2,
    )
    assert len(S.shape) == 3  # dPhys, dSlow, dSlow
    S_expanded = expand_S(S)
    return *make_S_square(L, S_expanded, R), info


def bell_original_threshold_keep_classical(
    As: MatrixStack, n_steps=500
) -> tuple[LeftSplittingTensor, BlockDiagTensor, RightSplittingTensor, dict]:
    tensor = utils.make_square(As, 2)
    L, S, R, info = bell_identical_blocks.combined_optimization(
        tensor,
        n_attempts_iterative=1,
        n_iterations_per_attempt=n_steps,
        early_stopping=False,
        maxiter_heuristic=0,
        keep_classical_correlations=True,
        tolEntropy=1e-2,
    )
    assert len(S.shape) == 3  # dPhys, dSlow, dSlow
    S_expanded = expand_S(S)
    return *make_S_square(L, S_expanded, R), info


def iterative_svd_micro_bsvd(
    As: MatrixStack, n_steps=500
) -> tuple[LeftSplittingTensor, BlockDiagTensor, RightSplittingTensor, dict]:
    L, M, R = vertical_svd.block_svd(As, n_iters=n_steps, initialize="bsvd", tolerance=0)
    # U_purified.shape =                     branch, L, l
    # blockdiag_purified.shape = branch, n_matrices, l, r
    # Vh_purified.shape =                    branch, r, R
    # LeftSplittingTensor =       Complex[np.ndarray, "nBranches dVirt dSlow"]
    # BlockDiagTensor =           Complex[np.ndarray, "dPhys nBranches dSlow dSlow"]
    # RightSplittingTensor =      Complex[np.ndarray, "nBranches dSlow dVirt"]
    S = rearrange(M, "b p l r -> p b l r")
    return *make_S_square(L, S, R), {}


def iterative_pulling_through_z0_dim_half(
    As: MatrixStack, n_steps=500
) -> tuple[LeftSplittingTensor, BlockDiagTensor, RightSplittingTensor, dict]:
    # return pulling_through.to_purification(As, svals_power = 0.0, equal_sized_blocks=True, verbose=False)
    kraus_L, kraus_R = pulling_through.find_simultaneous_kraus_operators(
        As, svals_power=0.0, max_iters=n_steps
    )
    # LPurifiedType = Complex[np.ndarray, "n_blocks dim_L dim_l"]
    # BDPurifiedType = Complex[np.ndarray, "n_blocks n_matrices dim_l dim_r"]
    # RPurifiedType = Complex[np.ndarray, "n_blocks dim_r dim_R"]
    # LeftSplittingTensor =       Complex[np.ndarray, "nBranches dVirt dSlow"]
    # BlockDiagTensor =           Complex[np.ndarray, "dPhys nBranches dSlow dSlow"]
    # RightSplittingTensor =      Complex[np.ndarray, "nBranches dSlow dVirt"]
    L, M, R = pulling_through.kraus_operators_to_LMR(As, kraus_L, kraus_R, equal_sized_blocks=True)
    S = rearrange(M, "b p l r -> p b l r")
    return *make_S_square(L, S, R), {}


############################################################################################################
# GRADIENT DESCENT METHODS
############################################################################################################


def rho_LM_MR_trace_norm_discard_classical_identical_blocks(  # aka graddesc_reconstruction_discard_classical_2
    As: MatrixStack,
    L: LeftSplittingTensor,
    S: BlockDiagTensor,
    R: RightSplittingTensor,
    n_steps=1000,
) -> PurificationMatrixStack:
    L, S, R = bell_identical_blocks.heuristic_optimization_LSR(
        As, L, S[:, 0], R, keep_classical_correlations=False, maxiter=n_steps
    )
    assert len(S.shape) == 3  # dPhys, dSlow, dSlow
    return LSR_to_purification(L, S, R, keep_classical=False)


def rho_LM_MR_trace_norm_identical_blocks(  # aka graddesc_reconstruction_keep_classical_2
    As: MatrixStack,
    L: LeftSplittingTensor,
    S: BlockDiagTensor,
    R: RightSplittingTensor,
    n_steps=1000,
) -> PurificationMatrixStack:
    L, S, R = bell_identical_blocks.heuristic_optimization_LSR(
        As, L, S[:, 0], R, keep_classical_correlations=True, maxiter=n_steps
    )
    assert len(S.shape) == 3  # dPhys, dSlow, dSlow
    return LSR_to_purification(L, S, R)


def rho_LM_MR_trace_norm(
    As: MatrixStack,
    L: LeftSplittingTensor,
    S: BlockDiagTensor,
    R: RightSplittingTensor,
    n_steps=1000,
) -> PurificationMatrixStack:
    L, S, R = bell_different_blocks.heuristic_optimization_LSR(As, L, S, R, maxiter=n_steps)
    assert len(S.shape) == 4  # dPhys, nBranches, dSlow, dSlow
    return LSR_to_purification(L, S, R)


# def rho_half_LM_MR_trace_norm(  # aka graddesc_reconstruction_2svals_rho_half
#     As: MatrixStack,
#     L: LeftSplittingTensor,
#     S: BlockDiagTensor,
#     R: RightSplittingTensor,
#     n_steps=1000,
# ) -> PurificationMatrixStack:
#     L, S, R = bmm_2svals_rho_half.heuristic_optimization_LSR(As, L, S, R, maxiter=n_steps)
#     assert len(S.shape) == 4  # dPhys, nBranches, dSlow, dSlow
#     return LSR_to_purification(L, S, R)


def graddesc_global_reconstruction_non_interfering(  # aka graddesc_non_interfering_local_reconstruction
    As: MatrixStack,
    L: LeftSplittingTensor,
    S: BlockDiagTensor,
    R: RightSplittingTensor,
    n_steps=1000,
) -> PurificationMatrixStack:
    tensors_orig = [As]
    tensors_a = [L[0:1], S[:, 0], R[0:1]]
    tensors_b = [L[1:], S[:, 1], R[1:]]

    tensors_a, tensors_b, info = graddesc_global.optimize(
        tensors_orig,
        tensors_a,
        tensors_b,
        interference_weight=1.0,
        epochs=n_steps // 50,
        steps_per_epoch=50,
    )
    theta_a = graddesc_global.contract_theta(tensors_a)
    theta_b = graddesc_global.contract_theta(tensors_b)
    return np.stack([theta_a, theta_b], axis=0)


def graddesc_global_reconstruction_split_non_interfering(  # aka graddesc_non_interfering_local_reconstruction_split
    As: MatrixStack,
    L: LeftSplittingTensor,
    S: BlockDiagTensor,
    R: RightSplittingTensor,
    n_steps=1000,
) -> PurificationMatrixStack:
    tensors_orig = graddesc_global.split(As)
    tensors_a = [L[0:1], *graddesc_global.split(S[:, 0]), R[0:1]]
    tensors_b = [L[1:], *graddesc_global.split(S[:, 1]), R[1:]]

    tensors_a, tensors_b, info = graddesc_global.optimize(
        tensors_orig,
        tensors_a,
        tensors_b,
        interference_weight=1.0,
        epochs=n_steps // 50,
        steps_per_epoch=50,
    )
    theta_a = graddesc_global.contract_theta(tensors_a)
    theta_b = graddesc_global.contract_theta(tensors_b)
    theta = np.stack([theta_a, theta_b], axis=0)
    return rearrange(theta, "b pa pb l r -> b (pa pb) l r")


def no_graddesc_identical_blocks_keep_classical(
    As: MatrixStack, L: LeftSplittingTensor, S: BlockDiagTensor, R: RightSplittingTensor, n_steps=0
) -> PurificationMatrixStack:
    return LSR_to_purification(L, S[:, 0], R, keep_classical=True)


def no_graddesc_identical_blocks_discard_classical(
    As: MatrixStack, L: LeftSplittingTensor, S: BlockDiagTensor, R: RightSplittingTensor, n_steps=0
) -> PurificationMatrixStack:
    return LSR_to_purification(L, S[:, 0], R, keep_classical=False)


def no_graddesc_different_blocks(
    As: MatrixStack, L: LeftSplittingTensor, S: BlockDiagTensor, R: RightSplittingTensor, n_steps=0
) -> PurificationMatrixStack:
    return LSR_to_purification(L, S, R, keep_classical=True)


############################################################################################################
# tolEntropy calculaion functions
############################################################################################################

# LeftSplittingTensor       : TypeAlias = Complex[NDArray, "nBranches dVirt_L dSlow"]
# BlockDiagTensor           : TypeAlias = Complex[NDArray, "dPhys nBranches dSlow dSlow"]
# RightSplittingTensor      : TypeAlias = Complex[NDArray, "nBranches dSlow dVirt_R"]
# MatrixStack               : TypeAlias = Complex[NDArray, "dPhys dVirt_L dVirt_R"]


def calculate_entropy_vertical(
    tensor: MatrixStack, L: LeftSplittingTensor, S: BlockDiagTensor, R: RightSplittingTensor
):
    """
    Calculate the residual vertical entanglement entropy after an attempted decomposision into
    block-diagonal form. Vertical entanglement entropy will be zero for exact blocks, where each
    block is identical. Zero vertical entanglement corresponds to a Bell-pair between L and R, not
    entangled with S. This is the entropy of rho_fast in https://arxiv.org/abs/2308.04291

    Inputs:
        tensor: MatrixStack: Complex[NDArray, "dPhys dVirt_L dVirt_R"]
            The original tensor which has attempted to be decomposed into L S R
            as tensor approx = einsum(L, S, R, "b L l, p b l r, b r R -> p L R")
            dPhys indexes the matrix in the stack. Each matrix is a dVirt_L x dVirt_R dimensional.

        L: LeftSplittingTensor: Complex[NDArray, "nBranches dVirt_L dSlow"]
            The (unitary) splitting matrix decomposing the tensor into a block-diagonal form from
            the left.

        S: BlockDiagTensor: Complex[NDArray, "dPhys nBranches dSlow dSlow"]
            The central block-diagonal stack of matrices, where dPhys indexes the matrix in the stack,
            nBranches indexes which block in the block-diagonal structure, and each block is of
            dimension dSlow x dSlow.

        R: RightSplittingTensor: Complex[NDArray, "nBranches dSlow dVirt_R"]
            The (unitary) splitting matrix decomposing the tensor into a block-diagonal form from
            the right.

    Output:
        entropy:
            The vertical entanglement entropy between the slow and fast degrees of freedom.
            This is zero for Bell-like entanglement (but nonzero for GHZ or non-branch-like entanglement)
    """
    # Assume L and R are unitary.
    #    L should be a unitary map from dVirt_L to nBranches x dSlow
    #    R should be a unitary map from dVirt_R to nBranches x dSlow

    L_mat = rearrange(L, "nBranches dVirt_L dSlow -> dVirt_L (nBranches dSlow)")
    uL_mat = utils.unitize(L_mat)
    uL = rearrange(uL_mat, "dVirt_L (nBranches dSlow) -> nBranches dVirt_L dSlow")

    R_mat = rearrange(R, "nBranches dSlow dVirt_R -> dVirt_R (nBranches dSlow)")
    uR_mat = utils.unitize(R_mat)
    uR = rearrange(uR_mat, "dVirt_R (nBranches dSlow) -> nBranches dSlow dVirt_R")

    # Compute the residual entanglement between slow and fast degrees of freedom
    overlap = einsum(np.conj(uL), tensor, np.conj(uR), "bl L l,  p L R,  br r R  ->  bl br p l r")
    rho_fast = einsum(overlap, np.conj(overlap), "bl br p l r,  blc brc p l r   ->  bl br blc brc")
    vertical_spectrum = np.linalg.eigvalsh(
        rearrange(rho_fast, "bl br blc brc -> (bl br) (blc brc)")
    )
    vertical_spectrum = vertical_spectrum / np.sum(vertical_spectrum)
    # Filter out zero eigenvalues to avoid log(0)
    vertical_spectrum = vertical_spectrum[vertical_spectrum > 0]
    vertical_entropy = -np.sum(np.log2(vertical_spectrum) * vertical_spectrum)
    return vertical_entropy


def calculate_entropy_cnot(
    tensor: MatrixStack, L: LeftSplittingTensor, S: BlockDiagTensor, R: RightSplittingTensor
):
    """
    Calculate the residual non-block entanglement entropy after-like entangement after an attempted decomposision into
    block-diagonal form. This entanglement entropy will be zero for exact blocks.
    Zero cnot entanglement corresponds to GHZ-like or Bell-pair between L and R, not
    entangled with S. This is the entropy of rho_fast in https://arxiv.org/abs/2308.04291

    Inputs:
        tensor: MatrixStack: Complex[NDArray, "dPhys dVirt_L dVirt_R"]
            The original tensor which has attempted to be decomposed into L S R
            as tensor approx = einsum(L, S, R, "b L l, p b l r, b r R -> p L R")
            dPhys indexes the matrix in the stack. Each matrix is a dVirt_L x dVirt_R dimensional.

        L: LeftSplittingTensor: Complex[NDArray, "nBranches dVirt_L dSlow"]
            The (unitary) splitting matrix decomposing the tensor into a block-diagonal form from
            the left.

        S: BlockDiagTensor: Complex[NDArray, "dPhys nBranches dSlow dSlow"]
            The central block-diagonal stack of matrices, where dPhys indexes the matrix in the stack,
            nBranches indexes which block in the block-diagonal structure, and each block is of
            dimension dSlow x dSlow.

        R: RightSplittingTensor: Complex[NDArray, "nBranches dSlow dVirt_R"]
            The (unitary) splitting matrix decomposing the tensor into a block-diagonal form from
            the right.

    Output:
        entropy:
            The vertical entanglement entropy between the slow and fast degrees of freedom.
            This is zero for Bell-like entanglement (but nonzero for GHZ or non-branch-like entanglement)
    """
    # Assume L and R are unitary.
    #    L should be a unitary map from dVirt_L to nBranches x dSlow
    #    R should be a unitary map from dVirt_R to nBranches x dSlow

    L_mat = rearrange(L, "nBranches dVirt_L dSlow -> dVirt_L (nBranches dSlow)")
    uL_mat = utils.unitize(L_mat)
    uL = rearrange(uL_mat, "dVirt_L (nBranches dSlow) -> nBranches dVirt_L dSlow")

    R_mat = rearrange(R, "nBranches dSlow dVirt_R -> dVirt_R (nBranches dSlow)")
    uR_mat = utils.unitize(R_mat)
    uR = rearrange(uR_mat, "dVirt_R (nBranches dSlow) -> nBranches dSlow dVirt_R")

    # Compute the residual entanglement between slow and fast degrees of freedom
    overlap = einsum(np.conj(uL), tensor, np.conj(uR), "bl L l,  p L R,  br r R  ->  bl br p l r")

    # A CNOT is composed of an XOR tensor and a COPY tensor. see https://arxiv.org/pdf/1708.00006
    xor_gate = np.zeros((2, 2, 2)) + 0.0j * np.zeros((2, 2, 2))
    xor_gate[0, 0, 0] += 1.0
    xor_gate[1, 1, 0] += 1.0
    xor_gate[0, 1, 1] += 1.0
    xor_gate[1, 0, 1] += 1.0

    xor_bottom_right = einsum(overlap, xor, "bl br p l r, bl br br_new -> bl br_new p l r")
    xor_bottom_left = einsum(overlap, xor, "bl br p l r, bl br bl_new -> bl_new br p l r")

    rho_bottom_right = einsum(
        xor_bottom_right, np.conj(xor_bottom_right), "bl br p l r, bl br p l r_prime -> r r_prime"
    )
    rho_bottom_left = einsum(
        xor_bottom_left, np.conj(xor_bottom_left), "bl br p l r, bl br p l_prime r -> l l_prime"
    )

    spectrum_bottom_right = np.linalg.eigvalsh(rho_bottom_right)
    spectrum_bottom_left = np.linalg.eigvalsh(rho_bottom_left)

    spectrum_bottom_right = spectrum_bottom_right / np.sum(spectrum_bottom_right)
    spectrum_bottom_left = spectrum_bottom_left / np.sum(spectrum_bottom_left)
    # Filter out zero eigenvalues to avoid log(0)
    spectrum_bottom_right = spectrum_bottom_right[spectrum_bottom_right > 0]
    spectrum_bottom_left = spectrum_bottom_left[spectrum_bottom_left > 0]

    entropy_bottom_right = -np.sum(np.log2(spectrum_bottom_right) * spectrum_bottom_right)
    entropy_bottom_left = -np.sum(np.log2(spectrum_bottom_left) * spectrum_bottom_left)

    return (entropy_bottom_right + entropy_bottom_left) / 2.0


############################################################################################################
# Combination function
############################################################################################################
def branch_from_theta(
    theta_scrambled: MatrixStack,
    iterative_method: None
    | Literal[
        "bell_discard_classical",
        "bell_keep_classical",
        "bell_original_threshold_discard_classical",
        "bell_original_threshold_keep_classical",
        "vertical_svd_micro_bsvd",
        "pulling_through",
    ],
    graddesc_method: None
    | Literal[
        "rho_LM_MR_trace_norm_discard_classical_identical_blocks",
        "rho_LM_MR_trace_norm_identical_blocks",
        "rho_LM_MR_trace_norm",
        # "rho_half_LM_MR_trace_norm",
        "graddesc_global_reconstruction_non_interfering",
        "graddesc_global_reconstruction_split_non_interfering",
    ],
    n_steps_iterative=500,
    n_steps_graddesc=1000,
    tolEntropy=None,
    tolEntropy_kind: Literal["cnot", "vertical"] | None = "cnot",
) -> tuple[PurificationMatrixStack, dict]:
    if iterative_method is None or iterative_method == "None":
        assert graddesc_method is None or graddesc_method == "None", (
            "Gradient descent without iterative initialization not supported"
        )
        return np.expand_dims(theta_scrambled, 0), {"iterative_time": 0.0, "graddesc_time": 0.0}

    if graddesc_method is not None:
        if "identical_blocks" in graddesc_method:
            if iterative_method not in ["bell_discard_classical", "bell_keep_classical"]:
                warnings.warn(
                    "The only iterative methods meant for identical blocks are 'bell_discard_classical' and 'bell_keep_classical'"
                )

    fn_dict_iterative = {
        "bell_discard_classical": bell_discard_classical,
        "bell_keep_classical": bell_keep_classical,
        "bell_original_threshold_discard_classical": bell_original_threshold_discard_classical,
        "bell_original_threshold_keep_classical": bell_original_threshold_keep_classical,
        "vertical_svd_micro_bsvd": iterative_svd_micro_bsvd,
        "pulling_through": iterative_pulling_through_z0_dim_half,
    }
    fn_iterative = fn_dict_iterative[iterative_method]

    if iterative_method == "bell_discard_classical":
        no_graddesc = no_graddesc_identical_blocks_discard_classical
    elif iterative_method == "bell_keep_classical":
        no_graddesc = no_graddesc_identical_blocks_keep_classical
    else:
        no_graddesc = no_graddesc_different_blocks

    fn_dict_graddesc = {
        None: no_graddesc,
        "rho_LM_MR_trace_norm_discard_classical_identical_blocks": rho_LM_MR_trace_norm_discard_classical_identical_blocks,
        "rho_LM_MR_trace_norm_identical_blocks": rho_LM_MR_trace_norm_identical_blocks,
        "rho_LM_MR_trace_norm": rho_LM_MR_trace_norm,
        # "rho_half_LM_MR_trace_norm": rho_half_LM_MR_trace_norm,
        "graddesc_global_reconstruction_non_interfering": graddesc_global_reconstruction_non_interfering,
        "graddesc_global_reconstruction_split_non_interfering": graddesc_global_reconstruction_split_non_interfering,
    }

    keep_classical = True
    if graddesc_method == "rho_LM_MR_trace_norm_discard_classical_identical_blocks":
        keep_classical = False
    if graddesc_method is None and "discard_classical" in iterative_method:
        keep_classical = False
    fn_graddesc = fn_dict_graddesc[graddesc_method]

    norm_orig = einsum(theta_scrambled, np.conj(theta_scrambled), "p l r, p l r -> ")
    tensor = utils.make_square(theta_scrambled, 2)

    t1 = time.time()
    L, S, R, info = fn_iterative(tensor, n_steps=n_steps_iterative)
    t2 = time.time()

    rejected = info.get("rejected"), False

    # Determine if the iterative method failed to find a good decomposition
    if tolEntropy is not None and tolEntropy_kind is not None:
        if tolEntropy_kind == "cnot":
            entropy = calculate_entropy_cnot(tensor, L, S, R)
        elif tolEntropy_kind == "vertical":
            entropy = calculate_entropy_vertical(tensor, L, S, R)
        else:
            assert False, f"unknown tolEntropy_kind {tolEntropy_kind}"

        info["entropy"] = entropy
        info["tolEntropy_kind"] = tolEntropy_kind

        if entropy > tolEntropy:
            info["rejected"] = True
            print(
                "    Further optimization was rejected as initial optimization failed to find a good decomposition."
            )
            print(f"    entropy = {entropy} (tolEntropy = {tolEntropy})")
            print(f"    tolEntropy_kind = {tolEntropy_kind}")

    # Only perform gradient descent steps if info["rejected"] == False
    theta_purified = fn_graddesc(
        tensor, L, S, R, n_steps=0 if info["rejected"] else n_steps_graddesc
    )
    t3 = time.time()
    return theta_purified, {"iterative_time": t2 - t1, "graddesc_time": t3 - t2, **info}


def branch(
    psi: MPS,
    iterative_method: None
    | Literal[
        "bell_discard_classical",
        "bell_keep_classical",
        "vertical_svd_micro_bsvd",
        "pulling_through",
    ],
    graddesc_method: None
    | Literal[
        "rho_LM_MR_trace_norm_discard_classical_identical_blocks",
        "rho_LM_MR_trace_norm_identical_blocks",
        "rho_LM_MR_trace_norm",
        # "rho_half_LM_MR_trace_norm",
        "graddesc_global_reconstruction_non_interfering",
        "graddesc_global_reconstruction_split_non_interfering",
    ],
    formL_target=1.0,
    formR_target=1.0,
    coarsegrain_from: int | Literal["half"] = "half",
    coarsegrain_size=2,
    n_steps_iterative=500,
    n_steps_graddesc=1000,
    tolEntropy=None,
    tolEntropy_kind: Literal["cnot", "vertical"] | None = "cnot",
) -> tuple[PurificationMatrixStack, dict]:
    if coarsegrain_from == "half":
        coarsegrain_from = int(psi.L / 2 - coarsegrain_size / 2)

    theta_scrambled = rearrange(
        psi.get_theta(
            coarsegrain_from, n=coarsegrain_size, formL=formL_target, formR=formR_target, cutoff=0.0
        ).to_ndarray(),
        "L p1 p2 R -> (p1 p2) L R",
    )

    return branch_from_theta(
        theta_scrambled,
        iterative_method,
        graddesc_method,
        n_steps_iterative=n_steps_iterative,
        n_steps_graddesc=n_steps_graddesc,
        tolEntropy=tolEntropy,
        tolEntropy_kind=tolEntropy_kind,
    )
