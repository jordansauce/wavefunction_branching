# %%
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Complex
from opt_einops import einsum, rearrange

import wavefunction_branching.decompositions.two_svds as bsvd
import wavefunction_branching.utils.tensors as tensorfns

Matrix = Complex[np.ndarray, "dim_L dim_R"]
MatrixStack = Complex[np.ndarray, "N_matrices dim_L dim_R"]
BDPurifiedType = Complex[np.ndarray, "which_block N_matrices dim_l dim_r"]
UPurifiedType = Complex[np.ndarray, "which_block dim_r dim_R"]
VhPurifiedType = Complex[np.ndarray, "which_block dim_L dim_l"]
SBD_RETURN_TYPE = tuple[Matrix, MatrixStack, Matrix, list[int]]

from wavefunction_branching.utils.tensors import unitize


def update_U_purified(
    A: MatrixStack, blockdiag_purified: BDPurifiedType, Vh_purified: VhPurifiedType
) -> UPurifiedType:
    """Do regular MERA-vidal-like optimization to find the unitary maximizing overlap with its environment"""
    env_U = einsum(np.conj(A), blockdiag_purified, Vh_purified, "p L R, b p l r, b r R -> L b l")
    L, b, l = env_U.shape
    env_U = rearrange(env_U, "L b l -> L (b l)")
    U_purified = np.conj(unitize(env_U))
    U_purified = rearrange(U_purified, "L (b l) -> b L l", b=b)
    return U_purified


def update_Vh_purified(
    A: MatrixStack, U_purified: UPurifiedType, blockdiag_purified: BDPurifiedType
) -> VhPurifiedType:
    """Do regular MERA-vidal-like optimization to find the unitary maximizing overlap with its environment"""
    env_Vh = einsum(np.conj(A), U_purified, blockdiag_purified, "p L R, b L l, b p l r -> b r R")
    b, r, R = env_Vh.shape
    env_Vh = rearrange(env_Vh, "b r R -> (b r) R")
    Vh_purified = np.conj(unitize(env_Vh))
    Vh_purified = rearrange(Vh_purified, "(b r) R -> b r R", b=b)
    return Vh_purified


def micro_bsvd(bottom):
    """Simultaneously block diagonalize a set of 2x2 matrices, by a unitary transformation"""
    p, l, r = bottom.shape
    assert l == 2, "micro-bsvd is desiged for nx2x2 tensors"
    assert r == 2, "micro-bsvd is desiged for nx2x2 tensors"
    ul, svals_l, _ = np.linalg.svd(rearrange(bottom, "s bl br -> bl (s br)"))
    _, svals_r, ur = np.linalg.svd(rearrange(bottom, "s bl br -> (s bl) br"))
    bottom_transformed = einsum(np.conj(ul), bottom, np.conj(ur), "bl l, s bl br, r br -> s l r")
    # Get the correct permutation
    bottom_norms = einsum(np.abs(bottom_transformed) ** 2, "s l r -> l r")
    if bottom_norms[0, 0] + bottom_norms[1, 1] < bottom_norms[1, 0] + bottom_norms[0, 1]:
        ul = ul[:, [1, 0]]
        bottom_transformed = bottom_transformed[:, [1, 0], :]
    return ul, bottom_transformed, ur


def update_blockdiag_purified(
    A: MatrixStack, U_purified: UPurifiedType, Vh_purified: VhPurifiedType
) -> tuple[UPurifiedType, BDPurifiedType, VhPurifiedType]:
    """Find the block-diagonal matrix maximizing overlap with its environment, by using a vertical SVD of the environment followed by a CP decomposition of the bottom half."""
    # Construct the environment of blockdiag_purified with A
    env = einsum(
        A, np.conj(U_purified), np.conj(Vh_purified), "p L R, bl L l, br r R -> bl br p l r"
    )
    bl, br, p, l, r = env.shape
    assert bl == br
    b = bl

    # Take the SVD of the environment "vertically"
    env = rearrange(env, "bl br p l r -> (bl br) (p l r)")
    bottom, vert_svals, top = np.linalg.svd(env, full_matrices=False)
    bottom = rearrange(bottom[:, :b], "(bl br) s -> s bl br", bl=bl, br=br)
    top = rearrange(top[:b, :], "s (p l r) -> s p l r", p=p, l=l, r=r)

    # Take a BSVD of the bottom
    bottom = einsum(bottom, vert_svals[:b], "s bl br, s -> s bl br")
    ul, bottom_transformed, ur = micro_bsvd(bottom)

    # Absorb the unitaries into U_purified, Vh_purified, and blockdiag_purified
    U_purified = einsum(U_purified, ul, "bl L l, bl b -> b L l")
    Vh_purified = einsum(Vh_purified, ur, "br r R, b br -> b r R")
    blockdiag_purified = einsum(top, bottom_transformed, "s p l r, s b b -> b p l r")
    return U_purified, blockdiag_purified, Vh_purified


def get_trace_norm_error(X: np.ndarray, Y: np.ndarray) -> float:
    """Returns 0 if As = U @ Bs @ Vh"""
    x = np.array(X.flatten())
    y = np.array(Y.flatten())
    x /= np.sqrt(np.conj(x) @ x)
    y /= np.sqrt(np.conj(y) @ y)
    trace_norm = np.sqrt(abs(1.0 - abs(np.conj(x) @ y) ** 2))
    assert isinstance(trace_norm, float)
    return trace_norm


def combine(
    U_purified: UPurifiedType, blockdiag_purified: BDPurifiedType, Vh_purified: VhPurifiedType
) -> MatrixStack:
    """Reconstruct the current best approximation to the original A MatrixStack"""
    return einsum(U_purified, blockdiag_purified, Vh_purified, "b L l, b p l r, b r R -> p L R")


def block_svd(
    A: MatrixStack,
    n_iters: int = 100,
    initialize: Literal["random", "bsvd"] = "bsvd",
    verbose: bool = False,
    tolerance: float = 1e-7,
    **kwargs,
) -> tuple[UPurifiedType, BDPurifiedType, VhPurifiedType]:
    """
    Inputs:
        A: the matrices to be simultaneously block-diagonalized (n_matrices dim_L dim_R)
    Outputs:
        U_purified            (n_branches dim_L dim_R_reduced)
        blockdiag_purified:   (n_branches n_matrices dim_L_reduced dim_R_reduced)
        Vh_purified:          (n_branches dim_R_reduced dim_R)
    such that A is approximated by einsum(U_purified, blockdiag_purified, Vh_purified, 'b L l, b p l r, b r R -> p L R')"""

    # Set the initial guesses for U_purified, blockdiag_purified, and Vh_purified
    if initialize == "bsvd":
        U, blockdiag, Vh, block_sizes = bsvd.block_svd(A, equal_sized_blocks=True, **kwargs)
        # U.shape =                     L, L
        # blockdiag.shape = n_matrices, L, R
        # Vh.shape =                    R, R
        U_purified, blockdiag_purified, Vh_purified = tensorfns.blockdiag_to_purification_split(
            U, blockdiag, Vh, block_sizes
        )
    else:
        U_purified = np.random.randn(2, A.shape[1], A.shape[1] // 2) + 1.0j * np.random.randn(
            2, A.shape[1], A.shape[1] // 2
        )
        blockdiag_purified = np.random.randn(
            2, A.shape[0], A.shape[1] // 2, A.shape[2] // 2
        ) + 1.0j * np.random.randn(2, A.shape[0], A.shape[1] // 2, A.shape[2] // 2)
        Vh_purified = np.random.randn(2, A.shape[2] // 2, A.shape[2]) + 1.0j * np.random.randn(
            2, A.shape[2] // 2, A.shape[2]
        )

    # U_purified.shape =                     branch, L, l
    # blockdiag_purified.shape = branch, n_matrices, l, r
    # Vh_purified.shape =                    branch, r, R

    errors = [get_trace_norm_error(A, combine(U_purified, blockdiag_purified, Vh_purified))]

    # Update U_purified, Vh_purified, and blockdiag_purified iteratively
    for i in range(n_iters):
        U_purified = update_U_purified(A, blockdiag_purified, Vh_purified)

        Vh_purified = update_Vh_purified(A, U_purified, blockdiag_purified)

        U_purified, blockdiag_purified, Vh_purified = update_blockdiag_purified(
            A, U_purified, Vh_purified
        )

        errors.append(get_trace_norm_error(A, combine(U_purified, blockdiag_purified, Vh_purified)))

        if i >= 10 and i % 10 == 0:
            if abs(errors[-1] - errors[-10]) < tolerance:
                break

    if verbose:
        plt.plot(errors, label="after blockdiag")
        plt.ylabel("Error")
        plt.xlabel("iteration step")
        plt.yscale("log")
        plt.legend()
        plt.show()

        print(errors)
    return U_purified, blockdiag_purified, Vh_purified


# import benchmarks.decompositions.block_diagonal.generate_test_inputs as gentest
# if __name__ == '__main__':
#     # block_sizes = [[15,20],[20,30]]
#     # block_sizes = [[22,20],[25,30]]
#     block_sizes = [25,25]
#     tensor = gentest.block_diagonal_matrices(block_sizes, noise_introduced=1e-5)
#     U, Vh = gentest.generate_scrambling_matrices('UV', tensor.shape[1], tensor.shape[2])
#     tensor_scrambled = einsum(U, tensor, Vh, 'l L, p L R, R r -> p l r')
#     U_purified, blockdiag_purified, Vh_purified = block_svd(tensor_scrambled, initialize='random', verbose=True)
