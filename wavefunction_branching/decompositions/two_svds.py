# # Simultaneous Block SVD:
# Decompose a set of matrices $\{A_i\}_{i=1}^K$ as

# $ A_i =  U B_i V^\dagger $

# Where each $A_i$ is an $m\times n$ matrix, each $B_i$ is a block-diagonal matrix with the same block-structure regardless of $i$, and $U$ and $V$ are $m\times m$ and $n \times n$ unitary matrices respectively.

# Or, in penrose graphical tensor notation:\
# ![A-UBU.png](https://lh3.googleusercontent.com/pw/ADCreHdSNajTy_la81rlU0u5gF9Oxbo_abjY71vUer64LLlnosB7JHAzfaiC3KF0lNMBYVtMFt3NKAdc7EzdI8xKJnbB7xGmSPuYXRnOkfx_g93XW8p6WNT8alENOco0s-vO58VTgYIv3jvhTTpBcnKJDAq6PQ=w1334-h102-s-no)

# such that
# ![A-B.png](https://lh3.googleusercontent.com/pw/ADCreHfiNdEoRR82_oSJpA4rffaWGXlWPgiBe7SxsinlKb5KWpBIRxVLRHdakkAS18hKbyAso3esnnOULztbTyAqQGr5G8D-yTLNT7sZUC7l5RytB-u6Sdlchu5rZkIgygqtCY5bMANAoHDAc3PKGXszYtvIxA=w1720-h511-s-no)

# If the input $A_i$ matrices commute, then each $B_i$ will be diagonal. If they do not, then each $B_i$ may still have some block-diagonal structure. This algorithm should find the minimal block-diagonal form, even in the presence of strong noise.

# For the theory, see ["Simultaneous singular value decomposition - Takanori Maehara and Kazuo Murota"](https://www.sciencedirect.com/science/article/pii/S0024379511000322)


from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Int, Num, Shaped
from opt_einops import (
    einsum,
    rearrange,
    repeat,
)  # Tensor opertations # ! pip install git+https://github.com/jordansauce/opt_einops

from wavefunction_branching.utils.plotting import vis, vprint


def permute_columns_for_block_diagonality_sorting(
    M: Shaped[Any, "... rows columns"],
) -> Int[np.ndarray, "columns"]:
    """Permutes the columns of a matrix to make it more block-diagonal.
    Inputs:
        M - a matrix to be permuted for more block-diagonality. Shape: (..., n_rows, n_columns)
    Outputs:
        column_perm_inds - list of integer indices
    Such that M_permuted = M[..., column_perm_inds]"""
    # Permute the columns of M to make it more block diagonal.
    # Takes a weighted-index-value approach to find the "ideal" indices for each column, then sorts them as close a possible to that.
    n_rows = M.shape[-2]
    n_columns = M.shape[-1]
    probs = M / repeat(np.sum(M, axis=-2), "... col ->  ... row col", row=n_rows)
    ideal_inds = np.sum(
        probs
        * repeat(np.arange(n_rows, dtype=probs.dtype), "... row ->  ... row col", col=n_columns),
        axis=-2,
    )
    column_perm_inds = np.argsort(ideal_inds)
    return column_perm_inds


def permute_rows_for_block_diagonality_sorting(
    M: Shaped[Any, "... rows columns"],
) -> Int[np.ndarray, "rows"]:
    """Just does a transpose before passing to permute_columns_for_block_diagonality()"""
    return permute_columns_for_block_diagonality_sorting(
        rearrange(M, "... rows columns -> ... columns rows")
    )


def permute_for_block_diagonality_asymmetric_graph_clustering(
    M: Shaped[Any, "... rows columns"], verbose=False
):  # _svecs
    X = abs(M)
    U_x, D_x, Vh_x = np.linalg.svd(X, full_matrices=False)

    sval_overlaps_L = X @ Vh_x[:2, :].conj().T
    perm_inds_L = np.argsort(abs(sval_overlaps_L[:, 0]) - abs(sval_overlaps_L[:, 1]))

    sval_overlaps_R = U_x[:, :2].conj().T @ X
    perm_inds_R = np.argsort(abs(sval_overlaps_R[0]) - abs(sval_overlaps_R[1]))

    if verbose:
        n = min(10, len(D_x))
        plt.scatter(np.arange(n), D_x[:n])
        plt.show()
        vprint("You should be able to see block diagonal structure in this:", verbose)
        svals_kept = 2
        reconstructed = einsum(
            U_x[:, :svals_kept], D_x[:svals_kept], Vh_x[:svals_kept, :], "l m, m, m r -> l r"
        )
        vis(reconstructed[perm_inds_L, :][:, perm_inds_R])
        vis(X[perm_inds_L, :][:, perm_inds_R])
        vprint("These should form well separated clusters:", verbose)
        plt.scatter(abs(sval_overlaps_L[:, 0]), abs(sval_overlaps_L[:, 1]))
        plt.scatter(abs(sval_overlaps_R[0]), abs(sval_overlaps_R[1]))
        plt.show()

    block_sizes = [
        np.sum(sval_overlaps_L[:, 0] > sval_overlaps_L[:, 1]),
        np.sum(sval_overlaps_L[:, 0] < sval_overlaps_L[:, 1]),
    ]
    return perm_inds_L, perm_inds_R, block_sizes


def permute_for_block_diagonality_symmetric_bigdiff(
    M: Shaped[Any, "... m m"], verbose=False
) -> tuple[Int[np.ndarray, "rows"], list[int]]:  # _bigdiff
    """Finds the same permutation on the rows and columns to make a matrix more block-diagonal.
        This function works by finding two rows with the biggest difference (sum(abs(row1)-abs(row2))), then building out blocks by finding similar vectors to those.
    Inputs:
        M - a square matrix to be permuted for more block-diagonality.
    Outputs:
        perm_inds - a list of indices such that M[:,perm_inds][perm_inds,:] is as block-diagonal as possible
        block_sizes - a list of the sizes of each block found"""
    X = abs(M) + abs(M.T)
    not_done_indices = set(np.arange(X.shape[0]))
    inds_in_blocks = [[], []]

    # Calculate X_difs[i,j] = np.sum(abs(X[i]-X[j])):
    Xr1 = repeat(X, "l r -> l1 l r", l1=X.shape[0])
    Xr2 = repeat(X, "l r -> l l2 r", l2=X.shape[0])
    X_difs = np.sum(abs(Xr1 - Xr2), axis=-1)
    # ^ This is a faster way of calculating X_difs[i,j] = np.sum(abs(X[i]-X[j])) than just looping over all emements
    ind_in_block_0, ind_in_block_1 = np.unravel_index(np.argmax(X_difs), X_difs.shape)
    inds_in_blocks[0] += [ind_in_block_0]
    inds_in_blocks[1] += [ind_in_block_1]
    not_done_indices.remove(ind_in_block_0)
    not_done_indices.remove(ind_in_block_1)
    block_0_candidate = X[ind_in_block_0] / np.linalg.norm(X[ind_in_block_0])
    block_1_candidate = X[ind_in_block_1] / np.linalg.norm(X[ind_in_block_1])

    while len(not_done_indices) > 0:
        overlaps_with_0 = einsum(
            block_0_candidate / np.linalg.norm(block_0_candidate),
            X[list(not_done_indices)],
            "r, l r -> l",
        )
        overlaps_with_1 = einsum(
            block_1_candidate / np.linalg.norm(block_1_candidate),
            X[list(not_done_indices)],
            "r, l r -> l",
        )
        overlaps_diff = overlaps_with_0 - overlaps_with_1
        ind_in_block_0 = np.argmax(overlaps_diff)
        ind_in_block_1 = np.argmin(overlaps_diff)
        if abs(overlaps_diff[ind_in_block_0]) > abs(overlaps_diff[ind_in_block_1]):
            ind_in_block_0 = list(not_done_indices)[ind_in_block_0]
            block_0_candidate += X[ind_in_block_0]
            inds_in_blocks[0] += [ind_in_block_0]
            not_done_indices.remove(ind_in_block_0)
        else:
            ind_in_block_1 = list(not_done_indices)[ind_in_block_1]
            block_1_candidate += X[ind_in_block_1]
            inds_in_blocks[1] += [ind_in_block_1]
            not_done_indices.remove(ind_in_block_1)
    if verbose:
        plt.plot(block_0_candidate[np.concatenate(inds_in_blocks)])
        plt.plot(block_1_candidate[np.concatenate(inds_in_blocks)])
        plt.show()
        vis(
            X[np.concatenate(inds_in_blocks), :][:, np.concatenate(inds_in_blocks)],
            figsize=(25, 25),
        )
    return np.concatenate(inds_in_blocks), [len(block) for block in inds_in_blocks]


def permute_rows_for_block_diagonality_bigdiff(
    M: Shaped[Any, "... rows columns"], equal_sized_blocks=False, verbose=False
) -> tuple[Int[np.ndarray, "rows"], list[int]]:
    """Finds the same permutation on the rows and columns to make a matrix more block-diagonal.
        This function works by finding two rows with the biggest difference (sum(abs(row1)-abs(row2))), then building out blocks by finding similar vectors to those.
    Inputs:
        M - a square matrix to be permuted for more block-diagonality.
    Outputs:
        perm_inds - a list of indices such that M[:,perm_inds][perm_inds,:] is as block-diagonal as possible
        block_sizes - a list of the sizes of each block found"""
    X = abs(M)
    not_done_indices = set(np.arange(X.shape[0]))
    inds_in_blocks = [[], []]

    # Find the vectors with the biggest difference (these should reside in different blocks)
    # Calculate X_difs[i,j] = np.sum(abs(X[i]-X[j])):
    Xr1 = repeat(X, "l r -> l1 l r", l1=X.shape[0])
    Xr2 = repeat(X, "l r -> l l2 r", l2=X.shape[0])
    X_difs = np.sum(abs(Xr1 - Xr2), axis=-1)
    # ^ This is a faster way of calculating X_difs[i,j] = np.sum(abs(X[i]-X[j])) than just looping over all emements
    ind_in_block_0, ind_in_block_1 = np.unravel_index(np.argmax(X_difs), X_difs.shape)
    avg_pos_0 = sum((X[ind_in_block_0] / sum(X[ind_in_block_0])) * np.arange(X.shape[1]))
    avg_pos_1 = sum((X[ind_in_block_1] / sum(X[ind_in_block_0])) * np.arange(X.shape[1]))
    vprint(f"avg_pos_0 = {avg_pos_0}, avg_pos_1 = {avg_pos_1}", verbose)
    if avg_pos_0 > avg_pos_1:  # Ensuring block-diagonality, rather than block anti-diagonality
        (ind_in_block_0, ind_in_block_1) = (ind_in_block_1, ind_in_block_0)
    inds_in_blocks[0] += [ind_in_block_0]
    inds_in_blocks[1] += [ind_in_block_1]
    not_done_indices.remove(ind_in_block_0)
    not_done_indices.remove(ind_in_block_1)
    block_0_candidate = X[ind_in_block_0] / np.linalg.norm(X[ind_in_block_0])
    block_1_candidate = X[ind_in_block_1] / np.linalg.norm(X[ind_in_block_1])

    # For each of the other vectors, assign them to one block or the other by similarity.
    while len(not_done_indices) > 0:
        overlaps_with_0 = einsum(
            block_0_candidate / np.linalg.norm(block_0_candidate),
            X[list(not_done_indices)],
            "r, l r -> l",
        )
        overlaps_with_1 = einsum(
            block_1_candidate / np.linalg.norm(block_1_candidate),
            X[list(not_done_indices)],
            "r, l r -> l",
        )
        overlaps_diff = overlaps_with_0 - overlaps_with_1
        ind_in_block_0 = np.argmax(overlaps_diff)
        ind_in_block_1 = np.argmin(overlaps_diff)

        expanding_0: bool = abs(overlaps_with_0[ind_in_block_0]) > abs(
            overlaps_with_1[ind_in_block_1]
        )

        if equal_sized_blocks:
            if len(inds_in_blocks[0]) >= X.shape[0] // 2:
                expanding_0 = False
            elif len(inds_in_blocks[1]) >= X.shape[0] // 2:
                expanding_0 = True

        if expanding_0:
            ind_in_block_0 = list(not_done_indices)[ind_in_block_0]
            block_0_candidate += X[ind_in_block_0]
            inds_in_blocks[0] += [ind_in_block_0]
            not_done_indices.remove(ind_in_block_0)
        else:
            ind_in_block_1 = list(not_done_indices)[ind_in_block_1]
            block_1_candidate += X[ind_in_block_1]
            inds_in_blocks[1] += [ind_in_block_1]
            not_done_indices.remove(ind_in_block_1)

    # assert set(inds_in_blocks[0]).union(set(inds_in_blocks[1])) == set(np.arange(X.shape[0]))
    if verbose:
        plt.plot(block_0_candidate[np.concatenate(inds_in_blocks)])
        plt.plot(block_1_candidate[np.concatenate(inds_in_blocks)])
        plt.show()
        # vis(X[np.concatenate(inds_in_blocks),:][:,np.concatenate(inds_in_blocks)], figsize = (5,5))
    return np.concatenate(inds_in_blocks), [len(block) for block in inds_in_blocks]


def permute_columns_for_block_diagonality_bigdiff(
    M: Shaped[Any, "... rows columns"], **kwargs
) -> tuple[Int[np.ndarray, "columns"], list[int]]:
    return permute_rows_for_block_diagonality_bigdiff(M.T, **kwargs)


def permute_for_block_diagonality_asymmetric_bigdiff(
    M: Shaped[Any, "... rows columns"], **kwargs
) -> tuple[Int[np.ndarray, "rows"], Int[np.ndarray, "columns"], list[int]]:
    perm_l, block_sizes_l = permute_rows_for_block_diagonality_bigdiff(M, **kwargs)
    perm_r, block_sizes_r = permute_columns_for_block_diagonality_bigdiff(M[perm_l, :], **kwargs)
    # B = M[perm_l,:][:,perm_r]
    return perm_l, perm_r, block_sizes_r


def random_linear_combination(
    A: Shaped[Any, "batch ..."], randvec: Shaped[Any, "batch"] | None = None
) -> Shaped[Any, "..."]:
    """Take a random linear combination over a batch of tensors
    Inputs:
        A - a batch of tensors - (batch ... )
        [optional] randvec - a vector of weights for the linear combination - (batch)
    Outputs:
        O - a tensor without the batch dimension - ( ... )
    """
    if randvec is None:
        randvec = np.random.normal(size=A.shape[0])
    return einsum(randvec, A, "batch, batch ... -> ...")


def get_block_diagonality_score(
    M: Shaped[Num, "rows columns"], block_sizes: list[int], norm_type: Literal["fro", "nuc"] = "fro"
) -> float:
    """Returns a measure of how block-diagonal a matrix is.
    Inputs:
        M - a matrix - (n_rows n_columns)
        block_sizes - a list of the size of each block-diagonal block
        norm_type - which norm to use to compare the norm_offblocks to the norm_onblocks - see np.linalg.norm()"""
    mask = np.ones_like(M)
    if block_sizes is not None:
        block_inds = np.insert(np.cumsum(block_sizes), 0, 0)
        for i in range(len(block_sizes)):
            mask[block_inds[i] : block_inds[i + 1], block_inds[i] : block_inds[i + 1]] = 0.0
    M_offblock = M * mask
    M_onblock = M * (1.0 - mask)
    norm_offblock = np.linalg.norm(M_offblock, ord=norm_type)
    norm_onblock = np.linalg.norm(M_onblock, ord=norm_type)
    ones_norm_offblock = np.linalg.norm(mask, ord=norm_type)
    ones_norm_onblock = np.linalg.norm(1.0 - mask, ord=norm_type)
    norm_onblock = norm_onblock / ones_norm_onblock
    norm_offblock = norm_offblock / ones_norm_offblock
    return float(norm_onblock / (norm_onblock + norm_offblock))


def unitize(U: Shaped[Num, "rows columns"]) -> Shaped[Num, "rows columns"]:
    """return a unitary matrix close to the input matrix"""
    UL, svals, UR = np.linalg.svd(U)
    return UL @ UR


def get_unitary_scale_amount(A: Shaped[Num, "batch rows columns"]) -> float:
    avg_norm_A = np.sum(np.sqrt(einsum(A, np.conj(A), "batch L R, batch L R -> batch")))
    unitary_scale = avg_norm_A / np.sqrt(A.shape[1])
    return unitary_scale


############################################################################################################################
def block_svd(
    A: Shaped[Num, "batch rows columns"],
    method: Literal["default", "fastest"] = "default",
    num_random_samples: int | None = None,
    verbose: bool = False,
    equal_sized_blocks: bool = False,
) -> tuple[
    Shaped[Num, "rows rows"],
    Shaped[Num, "batch rows columns"],
    Shaped[Num, "columns columns"],
    list[int],
]:
    """Simultaneously block-diagonalize a set of matrices in a SVD-like fashion: A[i] = U B[i] V†,
        where each B[i] is a block-diagonal matrix, and U and V are both unitary matrices.
        This algorithm will be tolerant to noise and give good results, but will be slower than approximate randomized versions. We reccomend using the randomized algorithm in the noiseless case.
    INPUTS:
        A: a stack of m×n matrices (batch, m, n) where m is the number of rows and n is the number of columns in each matrix A[i]
        method: either 'default' or 'fastest'
                'default' finds U and V by taking the SVD of the matrices in A concatenated into a big rectangular matrix.
                    - Reccomended most of the time, unless the input matrices are too big, or the batch size is too big (but in that case this method can be used with num_random_samples set less than the batch size)
                    - if num_random_samples is set, the batch size is reduced to num_random_samples by samping random linear combinations of the A matrices
                    - speed: O( batch_random_samples × (m^2 × n + n^2 × m) ) - or O( batch_size × (m^2 × n + n^2 × m) ) if batch_random_samples is None
                'fastest': Just take the SVD of a single random linear combination of the matrices in A.
                    - Fast, but will lead to worse results in the presence of noise, or if the blocks are not fully separable
                    - speed: min( O(m^2 × n), O(n^2 × m) )
    OUTPUTS:
        U, B, V, block_sizes :
        where U and V are unitary matrices, B is a set of block-diagonal matrices such that A[i] = U @ B[i] @ Vh
        U  - a unitary matrix with shape (m, m)
            such that A[i] @ A[i].T.conj() = U @ B[i] @ B[i].T.conj() @ U.T.conj()
        B  - a stack of block-diagonal matrices with the same shape as A: (batch, m, n).
            Each matrix B will have the same block-diagonal structure. B[i] = U.conj().T @ A[i] @ Vh.conj().T
        Vh - a unitary matrix with shape (n, n)
            such that A[i].T.conj() @ A[i] = Vh.T.conj() @ B[i].T.conj() @ B[i] @ Vh
        block_sizes: The dimension of each of the blocks found in B, if any.
    """

    if method == "default":
        A_to_svd = A
        # If running with the full batch size is too slow, we can make a reduced batch size out of random linear combinations of the original batch elements.
        if num_random_samples is not None:
            if num_random_samples < A.shape[0]:
                randweights = np.random.normal(size=(num_random_samples, A.shape[0]))
                A_to_svd = einsum(
                    randweights, A, "random_samples batch, batch ... -> random_samples ..."
                )

        # For U and V before permutation, take the SVD of the matrices in A by concatenating them into a big rectangular matrix
        # This involves two SVDs: one concatenating down the columns, and the other across the rows.
        U, D1, Vh_big = np.linalg.svd(
            rearrange(A_to_svd, "batch L R -> L (batch R)"), full_matrices=False
        )  # SVD(Concatenate across the rows, increasing the number of columns)
        U_big, D2, Vh = np.linalg.svd(
            rearrange(A_to_svd, "batch L R -> (batch L) R"), full_matrices=False
        )  # SVD(Concatenate down the columns, increasing the number of rows)
    elif method == "fastest":
        # Alternately, a U and a V can be found by taking the SVD of a random linear combination of the matrices in A.
        # This is faster, but will lead to worse results in the presence of noise, or if the blocks are not fully separable
        U, S, Vh = np.linalg.svd(random_linear_combination(A))
    else:
        raise ValueError("block_svd() 'method' input should be one of 'default' or 'fastest'")

    # Now we basically have the final matrices U and V. We just have to permute them to reveal the block structure
    B = U.conj().T @ A @ Vh.conj().T
    if verbose:
        vis_threshold = 0.0004 * abs(np.min(B))
        vis({"Before permutation, the block-diagonal tensor": B}, threshold=vis_threshold)
    # TODO: use the sum of squares rather than just the sum of absolutes
    row_perm_inds, column_perm_inds, block_sizes = permute_for_block_diagonality_asymmetric_bigdiff(
        np.sum(abs(B), axis=0), verbose=max(0, verbose - 1), equal_sized_blocks=equal_sized_blocks
    )
    U = U[:, row_perm_inds]
    Vh = Vh[column_perm_inds, :]
    B = U.conj().T @ A @ Vh.conj().T
    if verbose:
        vis({"The final block diagonal tensor": B}, threshold=vis_threshold)
        block_diagonality_score_final = get_block_diagonality_score(
            np.sum(abs(B), axis=0), block_sizes
        )
        print(f"The final block diagonality score = {block_diagonality_score_final}")
    return U, B, Vh, block_sizes


# ^^^ block_svd ^^^
############################################################################################################################
