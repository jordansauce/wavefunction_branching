import copy
from typing import Literal

import numpy as np
from jaxtyping import Complex
from opt_einops import einsum, rearrange

from wavefunction_branching.types import Matrix, MPSTensor


def make_square(tensor: MPSTensor, nBlocks: int = 2) -> MPSTensor:
    shape = list(tensor.shape)
    tensor = tensor.reshape(-1, shape[-2], shape[-1])  # physical, left, right
    # Ensure that the tensors are square
    if shape[-2] != shape[-1] or max(tensor.shape[1:]) % nBlocks != 0:
        shape = list(tensor.shape)
        chi = max(shape[1:])
        if chi % nBlocks != 0:  # Make the bond dimension divisible by the number of blocks
            chi += nBlocks - chi % nBlocks
        tensor_ = np.zeros((shape[0], chi, chi), dtype=tensor.dtype)
        tensor_[:, : shape[1], : shape[2]] = tensor
        tensor = tensor_
    return tensor


def unitize(matrix: Matrix) -> Matrix:
    """Return the closest isometric matrix to a given input matrix"""
    U, svals, Vh = np.linalg.svd(matrix)
    if Vh.shape[0] > U.shape[1]:
        Vh = Vh[: U.shape[1], :]
    if U.shape[1] > Vh.shape[0]:
        U = U[:, : Vh.shape[0]]
    return U @ Vh


def truncated_svd(
    matrix,
    absorb: tuple[float, float] | Literal["left", "right", "both"] | None = None,
    chi_max: int | None = None,
    svd_min: float = 0.0,
    trunc_cut: float = 0.0,
    relative=False,
):
    """Split a matrix (or batch of them) with an SVD, while optionally tuncating the bond dimension.
    Args:
        matrix: The matrix (or batch of matrices) to split with a truncated SVD (or SVDs)
            shape = (..., l, r)
        absorb: Whether to absorb the singular values into the left or right tensor, both, or None
        chi_max: The maximum bond dimension (number of singular values) to keep
        svd_min: Discard all singular values smaller than this
        trunc_cut: Discard all singular values as long as sum_{i discarded} svals[i]**2 <= trunc_cut**2
        relative: Whether trunc_cut and svd_min are relative or not (relative to the largest singular value in the case of svd_min, or relative to the sum of squares of all singular values, in the case of trunc_cut)"""
    # Split the tensor with an SVD
    L, svals, R = np.linalg.svd(matrix, full_matrices=False)
    # Truncate the singular values
    if trunc_cut > 0.0:
        trunc_cut = trunc_cut / np.linalg.norm(svals, axis=-1) if relative else trunc_cut
        svals[np.cumsum(svals[..., ::-1] ** 2, axis=-1)[..., ::-1] <= trunc_cut**2] = 0.0
    if svd_min > 0.0:
        svd_min = svd_min / svals[..., 0] if relative else svd_min
        svals[svals < svd_min] = 0.0
    n_nonzero_svals = np.max(np.sum(svals > 0, axis=-1))
    svals = svals[..., :n_nonzero_svals]
    if chi_max is not None and svals.shape[-1] > chi_max:
        svals = svals[..., :chi_max]

    L = L[..., :, : svals.shape[-1]]
    R = R[..., : svals.shape[-1], :]

    # Absorb the singular values into the left or right tensor if desired
    if absorb is None:
        return L, svals, R
    else:
        if absorb == "left":
            absorb = (1.0, 0.0)
        elif absorb == "right":
            absorb = (0.0, 1.0)
        elif absorb == "both":
            absorb = (0.5, 0.5)
        assert isinstance(absorb, tuple) and len(absorb) == 2
        return einsum(L, svals ** absorb[0], "... l m, m -> ... l m"), einsum(
            svals ** absorb[1], R, "m, ... m r -> ... m r"
        )


##############################^^^ truncated_svd() ^^^###################################


def truncate_tensor(tensor, pattern, **trunc_kwargs):
    """Truncate the singular values wthin a tensor to a given bond dimension.
    Args:
        tensor: The tensor (or batch of matrices) to split with trincated SVDs (shape = (..., l, r))
        pattern: The pattern specifying which indicies to split between, eg. pattern
            = 'a b c d -> (a b) (c d)' would truncate the singular values between (a b) and (c d)
        absorb: Whether to absorb the singular values into the left or right tensor, both, or None
        chi_max: The maximum bond dimension (number of singular values) to keep
        svd_min: The minimum size of singular values to keep
        trunc_cut: Discard all singular values as long as sum_{i discarded} svals[i]**2 <= trunc_cut**2 "
    Returns:
        tensor: The tensor with truncated singular values (same shape as input tensor)
        bond_dim: The number of singular values kept"""
    pattern_from = pattern.split("->")[0].strip()
    pattern_to = pattern.split("->")[1].strip()
    shapes = {key: tensor.shape[i] for i, key in enumerate(pattern_from.split(" "))}
    matrix = rearrange(tensor, f"{pattern_from} -> {pattern_to}")
    L, svals, R = truncated_svd(matrix, **trunc_kwargs)  # type: ignore
    # Put the tensor back into its original format
    tensor = einsum(L, svals, R, "l m, m, m r -> l r")
    return rearrange(tensor, f"{pattern_to} -> {pattern_from}", **shapes), svals.shape[-1]


##############################^^^ truncate_tensor() ^^^###################################


def blockdiag_to_purification_split(
    U: Complex[np.ndarray, "L L"],
    theta_blockdiag: Complex[np.ndarray, "p L R"],
    Vh: Complex[np.ndarray, "R R"],
    block_sizes_L: list[int],
    block_sizes_R: list[int] | None = None,
) -> tuple[
    Complex[np.ndarray, "b L l"], Complex[np.ndarray, "b p l r"], Complex[np.ndarray, "b r R"]
]:  # theta_purified[i] = U @ theta_block_i @ Vh
    """
    Inputs:
        U:                Complex[np.ndarray, "L L"]:                chi * chi
        theta_blockdiag:  Complex[np.ndarray, "p L R"]:   physical * chi * chi
        Vh:               Complex[np.ndarray, "R R"],                chi * chi
    Outputs:
        U_purified:       Complex[np.ndarray, "b L l"]:     n_branches * chi * chi_reduced
        theta_purified:   Complex[np.ndarray, "b p l r"]:   n_branches * physical * chi_reduced * chi_reduced
        Vh_purified:      Complex[np.ndarray, "b r R"],     n_branches * chi_reduced * chi
    """
    if block_sizes_R is None:
        block_sizes_R = block_sizes_L
    assert len(block_sizes_L) == len(block_sizes_R)

    old_dim_L = theta_blockdiag.shape[1]
    old_dim_R = theta_blockdiag.shape[2]

    if np.sum(block_sizes_L) != old_dim_L:
        block_sizes = copy.deepcopy(block_sizes_L)
        extra = old_dim_L - np.sum(block_sizes)
        block_sizes_L = []
        for i in range(len(block_sizes)):
            block_sizes_L.append(block_sizes[i] + extra // len(block_sizes))
        if np.sum(block_sizes_L) != old_dim_L:
            block_sizes_L[-1] += old_dim_L - np.sum(block_sizes_L)

    if np.sum(block_sizes_R) != old_dim_R:
        block_sizes = copy.deepcopy(block_sizes_R)
        extra = old_dim_R - np.sum(block_sizes)
        block_sizes_R = []
        for i in range(len(block_sizes)):
            block_sizes_R.append(block_sizes[i] + extra // len(block_sizes))
        if np.sum(block_sizes_R) != old_dim_R:
            block_sizes_R[-1] += old_dim_R - np.sum(block_sizes_R)

    old_dim_L = sum(block_sizes_L)
    new_dim_L = max(block_sizes_L)
    old_dim_R = sum(block_sizes_R)
    new_dim_R = max(block_sizes_R)
    n_branches = len(block_sizes_L)

    assert old_dim_L == theta_blockdiag.shape[1]
    assert old_dim_R == theta_blockdiag.shape[2]

    p = theta_blockdiag.shape[0]

    block_inds_L = [0] + np.cumsum(block_sizes_L).tolist() + [np.sum(block_sizes_L)]
    block_inds_R = [0] + np.cumsum(block_sizes_R).tolist() + [np.sum(block_sizes_R)]

    for i in range(n_branches):
        assert block_inds_L[i + 1] - block_inds_L[i] == block_sizes_L[i]
        assert block_inds_R[i + 1] - block_inds_R[i] == block_sizes_R[i]

    # Purify the block diagonal tensor
    theta_purified = np.zeros((n_branches, p, new_dim_L, new_dim_R), dtype=theta_blockdiag.dtype)
    for i in range(n_branches):
        theta_purified[i, :, : block_sizes_L[i], : block_sizes_R[i]] = theta_blockdiag[
            :, block_inds_L[i] : block_inds_L[i + 1], block_inds_R[i] : block_inds_R[i + 1]
        ]

    # Purify U
    U_purified = np.zeros((n_branches, old_dim_L, new_dim_L), dtype=U.dtype)
    for i in range(n_branches):
        U_purified[i, :, : block_sizes_L[i]] = U[:, block_inds_L[i] : block_inds_L[i + 1]]

    # Purify Vh
    Vh_purified = np.zeros((n_branches, new_dim_R, old_dim_R), dtype=Vh.dtype)
    for i in range(n_branches):
        Vh_purified[i, : block_sizes_R[i], :] = Vh[block_inds_R[i] : block_inds_R[i + 1], :]

    return U_purified, theta_purified, Vh_purified


##############################^^^ blockdiag_to_purification_split() ^^^###################################


def blockdiag_to_purification(
    U: Complex[np.ndarray, "L L"],
    theta_blockdiag: Complex[np.ndarray, "p L R"],
    Vh: Complex[np.ndarray, "R R"],
    block_sizes,
) -> Complex[np.ndarray, "branch p L R"]:  # theta_purified[i] = U @ theta_block_i @ Vh
    """
    Inputs:
        U:              Complex[np.ndarray, "L L"]:                chi * chi
        theta_blockdiag:   Complex[np.ndarray, "p L R"]:   physical * chi * chi
        Vh:             Complex[np.ndarray, "R R"],                chi * chi
    Outputs:
        theta_purified: Complex[np.ndarray, "branch p l r"],    n_branches * physical * chi * chi
    """
    U_purified, theta_purified, Vh_purified = blockdiag_to_purification_split(
        U, theta_blockdiag, Vh, block_sizes
    )

    return einsum(
        U_purified,
        theta_purified,
        Vh_purified,
        "branch L l, branch p l r, branch r R -> branch p L R",
    )


##############################^^^ blockdiag_to_purification() ^^^###################################
