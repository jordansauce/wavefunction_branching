# %%
import copy
from functools import partial
from tabnanny import verbose
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Complex

# import wavefunction_branching.decompositions.block_diagonal.simultaneous_block_svd as bsvd
# import wavefunction_branching.utils.tensors as tensorfns
from opt_einops import einsum, rearrange
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity
from tqdm.autonotebook import tqdm

from wavefunction_branching.utils.plotting import vis

Matrix = Complex[np.ndarray, "dim_L dim_R"]
LLMatrix = Complex[np.ndarray, "dim_L dim_L"]
RRMatrix = Complex[np.ndarray, "dim_R dim_R"]
MatrixStack = Complex[np.ndarray, "n_matrices dim_L dim_R"]
BDPurifiedType = Complex[np.ndarray, "n_blocks n_matrices dim_l dim_r"]
LPurifiedType = Complex[np.ndarray, "n_blocks dim_L dim_l"]
RPurifiedType = Complex[np.ndarray, "n_blocks dim_r dim_R"]
SBD_RETURN_TYPE = tuple[Matrix, MatrixStack, Matrix, list[int]]


def make_positive_sqrt(matrix, normalize_spectrum=True):
    B = matrix @ matrix.conj().T
    eigval, eigvec = np.linalg.eigh(B)
    # eigval[eigval < 0] *= 0.5
    if normalize_spectrum:
        eigval -= eigval[0]
        eigval /= eigval[-1]
    eigval = np.abs(eigval) ** 0.5
    return eigvec.dot(np.diag(eigval)).dot(eigvec.conj().T)


def make_positive_sum(matrix, normalize_spectrum=True):
    """Finds a positive-semidefinite matrix close to the input matrix, but not necessarily the closest (unless the input matrix is already symmetric)"""
    B = (matrix + matrix.conj().T) / 2
    eigval, eigvec = np.linalg.eigh(B)
    eigval[eigval < 0] = 0
    if normalize_spectrum:
        eigval -= eigval[0]
        eigval /= eigval[-1]
    return eigvec.dot(np.diag(eigval)).dot(eigvec.conj().T)


def make_positive_closest(matrix, normalize_spectrum=True):  # -> Any:
    """Find the nearest positive-semidefinite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATlaB code [1], which
    credits [2].
    https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (matrix + matrix.conj().T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.conj().T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.conj().T) / 2

    if is_positive(A3):
        return make_positive_sum(A3, normalize_spectrum=normalize_spectrum)

    spacing = np.spacing(np.linalg.norm(matrix))
    # The above is different from [1]. It appears that MATlaB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(matrix.shape[0])
    k = 1
    while not is_positive(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return make_positive_sum(A3, normalize_spectrum=normalize_spectrum)


def is_positive(matrix):
    """Returns true when input matrix is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def normalize_spectrum(matrix):
    """Return a modified matrix where the singular values lie between zero and one"""


def find_simultaneous_kraus_operators(
    matrix_stack: MatrixStack,
    svals_power: float = 0.0,
    normalize_spectrum: bool = True,
    make_positive_method: Literal["sqrt", "sum", "closest", "none"] = "sqrt",
    max_iters: int = 1000,
    convergence_threshold: float = 1e-5,
    verbose=0,
) -> tuple[LLMatrix, RRMatrix]:
    """
    Inputs:
        matrix_stack (n_matrices dim_L dim_R): the matrices to find simultaneous kraus operators for
        svals_power (float): the power to which to raise the singular values of matrix_stack

    Outputs: Tuple[kraus_L, kraus_r]:
        kraus_L (dim_L dim_L): a positive matrix
        kraus_R (dim_R dim_R): a positive matrix

    such that    kraus_L @ matrix_stack[i]  =  matrix_stack[i] @ kraus_R
    """
    n_matrices, dim_L, dim_R = matrix_stack.shape

    # initialize kraus operators
    kraus_L = np.random.randn(dim_L, dim_L) + 1.0j * np.random.randn(dim_L, dim_L)
    make_positive_methods = {
        "sqrt": partial(make_positive_sqrt, normalize_spectrum=normalize_spectrum),
        "sum": partial(make_positive_sum, normalize_spectrum=normalize_spectrum),
        "closest": partial(make_positive_closest, normalize_spectrum=normalize_spectrum),
        "none": lambda x: x,
    }
    assert (
        make_positive_method in make_positive_methods
    ), f"make_positive_method must be one of {list(make_positive_methods.keys())}"
    make_positive_fn = make_positive_methods[make_positive_method]
    kraus_L = make_positive_fn(kraus_L)

    # Pre-compute the SVDs and inverses of C as ((n_matrices dim_L) | dim_R) and (dim_L | (n_matrices dim_R))
    matrix_stack_l_nr = rearrange(
        matrix_stack, "n_matrices dim_L dim_R -> dim_L (n_matrices dim_R)"
    )
    matrix_stack_nl_r = rearrange(
        matrix_stack, "n_matrices dim_L dim_R -> (n_matrices dim_L) dim_R"
    )

    U_l, svals_l_nr, Vh_nr = np.linalg.svd(matrix_stack_l_nr, full_matrices=False)
    U_nl, svals_nl_r, Vh_r = np.linalg.svd(matrix_stack_nl_r, full_matrices=False)

    svals_l_nr = svals_l_nr**svals_power
    svals_nl_r = svals_nl_r**svals_power

    C_l_nr = U_l @ np.diag(svals_l_nr) @ Vh_nr
    C_l_nr_inv = np.conj(U_l) @ np.diag(svals_l_nr**-1) @ np.conj(Vh_nr)
    C_nl_r = U_nl @ np.diag(svals_nl_r) @ Vh_r
    C_nl_r_inv = np.conj(U_nl) @ np.diag(svals_nl_r**-1) @ np.conj(Vh_r)

    assert np.allclose(C_l_nr @ C_l_nr_inv.T, np.eye(dim_L))
    assert np.allclose(C_nl_r_inv.T @ C_nl_r, np.eye(dim_R))

    C_l_nr = rearrange(
        C_l_nr, "dim_L (n_matrices dim_R) ->  n_matrices dim_L dim_R", n_matrices=n_matrices
    )
    C_l_nr_inv = rearrange(
        C_l_nr_inv, "dim_L (n_matrices dim_R) ->  n_matrices dim_L dim_R", n_matrices=n_matrices
    )
    C_nl_r = rearrange(
        C_nl_r, "(n_matrices dim_L) dim_R ->  n_matrices dim_L dim_R", n_matrices=n_matrices
    )
    C_nl_r_inv = rearrange(
        C_nl_r_inv, "(n_matrices dim_L) dim_R ->  n_matrices dim_L dim_R", n_matrices=n_matrices
    )

    # Perform the iterative updates to the krauss pulling through operators
    delta_Rs = []
    svals_Rs = []
    i_plotted = []
    kraus_R = einsum(C_nl_r_inv, kraus_L, C_nl_r, "n L1 R1,     L1 L2,   n L2 R2 ->  R1 R2")
    kraus_R = make_positive_fn(kraus_R)
    pbar = tqdm(total=-np.log10(convergence_threshold))
    for i in range(max_iters):
        kraus_R_old = copy.deepcopy(kraus_R)

        # Update L
        kraus_L = einsum(C_l_nr, kraus_R, C_l_nr_inv, "n L1 R1,  R1 R2,    n L2 R2     ->  L1 L2")
        kraus_L = make_positive_fn(kraus_L)

        # Update R
        kraus_R = einsum(C_nl_r_inv, kraus_L, C_nl_r, "n L1 R1,     L1 L2,   n L2 R2 ->  R1 R2")
        kraus_R = make_positive_fn(kraus_R)

        # Check for convergence
        delta_R = np.linalg.norm(np.abs(kraus_R_old - kraus_R), ord="fro")
        delta_Rs.append(delta_R)

        # Update the progress bar
        new_progress = -np.log10(delta_R)
        if new_progress > pbar.n:
            pbar.update(new_progress - pbar.n)
            pbar.refresh()

        # Break if converged
        if delta_R < convergence_threshold:
            print(f"Converged after {i} iterations")
            break
        if len(delta_Rs) > 50:
            if abs(delta_R - np.mean(delta_Rs[-50:])) < 0.1 * convergence_threshold:
                print(f"Answer stopped improving after {i} iterations")
                break

        if verbose > 1:
            if i % 20 == 0:
                try:
                    svals_R = np.linalg.svd(kraus_R, compute_uv=False)
                    svals_Rs.append(svals_R)
                    i_plotted.append(i)
                except np.linalg.LinAlgError as e:
                    print(f"Error in taking the SVD when plotting at iteration {i}: {e}")

    if len(i_plotted) > 1:
        plt.plot(delta_Rs)
        plt.yscale("log")
        plt.ylabel("Change in the norm of the right kraus operator")
        plt.xlabel("iteration step")
        plt.show()

        plt.plot(
            i_plotted,
            np.array(svals_Rs) * (1.0 - 0.1 * convergence_threshold) + 0.1 * convergence_threshold,
        )
        plt.yscale(
            "logit",
        )
        plt.grid()
        plt.ylabel("Singular values of the right kraus operator")
        plt.xlabel("iteration step")
        plt.show()

    if verbose:
        try:
            svals_L = np.linalg.svd(kraus_L, compute_uv=False)
            svals_R = np.linalg.svd(kraus_R, compute_uv=False)
        except np.linalg.LinAlgError:
            print("Error in taking the SVD when plotting at the end")
        else:
            plt.plot(svals_L, label="svals_L")
            plt.plot(svals_R, label="svals_R")
            plt.plot(np.sqrt(abs(1.0 - svals_L**2)), label="sqrt(1 - svals_L^2)", linestyle="--")
            plt.plot(np.sqrt(abs(1.0 - svals_R**2)), label="sqrt(1 - svals_R^2)", linestyle="--")
            plt.ylabel("Singular value magintude")
            plt.xlabel("Singular value index")
            plt.legend()
            plt.show()

    pbar.close()

    return kraus_L, kraus_R


def kraus_operators_to_purification_non_overlapping(
    matrix_stack: MatrixStack,
    kraus_L: LLMatrix,
    kraus_R: RRMatrix,
    equal_sized_blocks: bool = False,
) -> BDPurifiedType:
    evals_L, Uh = np.linalg.eigh(kraus_L)
    evals_R, Vh = np.linalg.eigh(kraus_R)

    assert np.allclose(Uh @ np.diag(np.abs(evals_L)) @ Uh.conj().T, kraus_L)
    assert np.allclose(Vh @ np.diag(np.abs(evals_R)) @ Vh.conj().T, kraus_R)

    # print(f'evals_L = {evals_L}')
    # print(f'evals_R = {evals_R}')

    C_transformed = einsum(Uh.conj().T, matrix_stack, Vh, "l L, p L R, R r -> p l r")
    if verbose:
        vis(C_transformed, name="matrix_stack transformed into kraus basis")

    if equal_sized_blocks:
        median_L = np.median(evals_L)
        median_R = np.median(evals_R)
    else:
        median_L = 0.5
        median_R = 0.5

    block_sizes_L = [np.sum(evals_L > median_L), np.sum(evals_L <= median_L)]
    block_sizes_R = [np.sum(evals_R > median_R), np.sum(evals_R <= median_R)]

    # print(f'block_sizes_L = {block_sizes_L}, \nblock_sizes_R = {block_sizes_R}')
    C_block_0_transformed = einsum(
        C_transformed, evals_L > median_L, evals_R > median_R, "p L R, L, R -> p L R"
    )
    C_block_1_transformed = einsum(
        C_transformed, evals_L <= median_L, evals_R <= median_R, "p L R, L, R -> p L R"
    )

    C_purif_transformed = np.stack([C_block_0_transformed, C_block_1_transformed])

    # Convert back to the original basis
    C_purif_transformed = einsum(
        Uh, C_purif_transformed, Vh.conj().T, "l L, b p L R, R r -> b p l r"
    )
    return C_purif_transformed


def find_best_cut(vector, verbose=False):
    X = vector.reshape(-1, 1)
    kde = KernelDensity(kernel="gaussian", bandwidth=0.03).fit(X)
    s = np.linspace(0, 1, 70)
    e = kde.score_samples(s.reshape(-1, 1))
    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    if len(mi) == 0:
        cut = 0.0
    else:
        cut = s[mi[np.argmin([e[i] for i in mi])]]
    if verbose:
        plt.plot(s, e)
        plt.vlines(cut, min(e), max(e))
        plt.scatter(X, np.random.random(X.shape[0]) * (max(e) - min(e)) + min(e))
        plt.show()
    return cut


def kraus_operators_to_LMR(
    matrix_stack: MatrixStack,
    kraus_L: LLMatrix,
    kraus_R: RRMatrix,
    branches_dim_factor: float = 0.5,
    threshold=1e-12,
    equal_sized_blocks=False,
    verbose=True,
) -> tuple[LPurifiedType, BDPurifiedType, RPurifiedType]:
    evals_L, Uh = np.linalg.eigh(kraus_L)
    evals_R, Vh = np.linalg.eigh(kraus_R)

    assert np.allclose(Uh @ np.diag(np.abs(evals_L)) @ Uh.conj().T, kraus_L)
    assert np.allclose(Vh @ np.diag(np.abs(evals_R)) @ Vh.conj().T, kraus_R)

    # dim_L_block_0 = int(np.ceil(len(evals_L) * branches_dim_factor))
    # dim_L_block_1 = int(np.floor(len(evals_L) * branches_dim_factor))

    # dim_R_block_0 = int(np.ceil(len(evals_R) * branches_dim_factor))
    # dim_R_block_1 = int(np.floor(len(evals_R) * branches_dim_factor))

    evals_L_block_0 = copy.deepcopy(evals_L)
    evals_L_block_1 = copy.deepcopy(evals_L)
    evals_R_block_0 = copy.deepcopy(evals_R)
    evals_R_block_1 = copy.deepcopy(evals_R)

    if equal_sized_blocks:
        cut_L = np.median(evals_L_block_0)
        cut_R = np.median(evals_L_block_0)
    else:
        # Find the best place to cut the evals_L_block_0
        cut_L = find_best_cut(evals_L_block_0, verbose=verbose)
        cut_R = find_best_cut(evals_R_block_0, verbose=verbose)

    evals_L_block_0[evals_L_block_0 > cut_L] = 1
    evals_L_block_0[evals_L_block_0 <= cut_L] = 0
    evals_R_block_0[evals_R_block_0 > cut_R] = 1
    evals_R_block_0[evals_R_block_0 <= cut_R] = 0
    evals_L_block_1 = np.sqrt(abs(1.0 - evals_L_block_0**2))
    evals_R_block_1 = np.sqrt(abs(1.0 - evals_R_block_0**2))

    # if equal_sized_blocks:
    #     evals_L_block_0[dim_L_block_0:] = 1
    #     evals_L_block_0[:len(evals_L) - dim_L_block_1] = 0
    #     evals_L_block_1 = np.sqrt(abs(1.0 - evals_L_block_0**2))

    #     evals_R_block_0[dim_R_block_0:] = 1
    #     evals_R_block_0[:len(evals_R) - dim_R_block_1] = 0
    #     evals_R_block_1 = np.sqrt(abs(1.0 - evals_R_block_0**2))

    C_transformed = einsum(Uh.conj().T, matrix_stack, Vh, "l L, p L R, R r -> p l r")
    C_block_0_transformed = einsum(
        C_transformed, evals_L_block_0**0.5, evals_R_block_0**0.5, "p L R, L, R -> p L R"
    )
    C_block_1_transformed = einsum(
        C_transformed, evals_L_block_1**0.5, evals_R_block_1**0.5, "p L R, L, R -> p L R"
    )

    Uh_block_0 = Uh[:, evals_L_block_0 > threshold]
    V_block_0 = (Vh[:, evals_R_block_0 > threshold]).conj().T
    C_block_0_transformed = C_block_0_transformed[:, evals_L_block_0 > threshold, :][
        :, :, evals_R_block_0 > threshold
    ]
    Uh_block_1 = Uh[:, evals_L_block_1 > threshold]
    V_block_1 = (Vh[:, evals_R_block_1 > threshold]).conj().T
    C_block_1_transformed = C_block_1_transformed[:, evals_L_block_1 > threshold, :][
        :, :, evals_R_block_1 > threshold
    ]

    # print(f'Uh_block_0.shape            = {Uh_block_0.shape}')
    # print(f'V_block_0.shape            = {V_block_0.shape}')
    # print(f'C_block_0_transformed.shape = {C_block_0_transformed.shape}')
    # print(f'Uh_block_1.shape            = {Uh_block_1.shape}')
    # print(f'V_block_1.shape            = {V_block_1.shape}')
    # print(f'C_block_1_transformed.shape = {C_block_1_transformed.shape}')

    # Put the blocks into a single tensor, indexed by a "which block" index out front
    C_purif_transformed_shape = np.insert(
        np.max(np.array([C_block_0_transformed.shape, C_block_1_transformed.shape]), axis=0), 0, 2
    )
    L_purif_shape = np.insert(np.max(np.array([Uh_block_0.shape, Uh_block_1.shape]), axis=0), 0, 2)
    R_purif_shape = np.insert(np.max(np.array([V_block_0.shape, V_block_1.shape]), axis=0), 0, 2)
    C_purif_transformed = np.zeros(C_purif_transformed_shape, dtype=C_block_0_transformed.dtype)
    L_purif = np.zeros(L_purif_shape, dtype=Uh_block_0.dtype)
    R_purif = np.zeros(R_purif_shape, dtype=V_block_0.dtype)

    C_purif_transformed[
        0,
        : C_block_0_transformed.shape[0],
        : C_block_0_transformed.shape[1],
        : C_block_0_transformed.shape[2],
    ] = C_block_0_transformed
    C_purif_transformed[
        1,
        : C_block_1_transformed.shape[0],
        : C_block_1_transformed.shape[1],
        : C_block_1_transformed.shape[2],
    ] = C_block_1_transformed

    L_purif[0, : Uh_block_0.shape[0], : Uh_block_0.shape[1]] = Uh_block_0
    L_purif[1, : Uh_block_1.shape[0], : Uh_block_1.shape[1]] = Uh_block_1

    R_purif[0, : V_block_0.shape[0], : V_block_0.shape[1]] = V_block_0
    R_purif[1, : V_block_1.shape[0], : V_block_1.shape[1]] = V_block_1

    return L_purif, C_purif_transformed, R_purif


def LMR_to_purification(L, M, R):
    return einsum(L, M, R, "b L l, b p l r, b r R -> b p L R")


def to_purification(
    matrix_stack: MatrixStack,
    equal_sized_blocks: bool = False,
    branches_dim_factor: float = 0.5,
    **kwargs,
):
    kraus_L, kraus_R = find_simultaneous_kraus_operators(matrix_stack, **kwargs)

    return LMR_to_purification(
        *kraus_operators_to_LMR(
            matrix_stack,
            kraus_L,
            kraus_R,
            branches_dim_factor=branches_dim_factor,
            equal_sized_blocks=equal_sized_blocks,
        )
    )


# import benchmarks.decompositions.block_diagonal.generate_test_inputs as gentest
# if __name__ == '__main__':
#     # block_sizes = [[50,30], [15,20]]
#     block_sizes = [20, 20, 20]
#     tensor = gentest.block_diagonal_matrices(block_sizes, noise_introduced=1e-2)
#     U, Vh = gentest.generate_scrambling_matrices('UV', tensor.shape[1], tensor.shape[2])
#     vis(tensor)
#     tensor_scrambled = einsum(U, tensor, Vh, 'l L, p L R, R r -> p l r')
#     vis(tensor_scrambled)
#     kraus_L, kraus_R = find_simultaneous_kraus_operators(tensor_scrambled, svals_power=0.0, make_positive_method="sqrt")


#     purification =  LMR_to_purification(*kraus_operators_to_LMR(tensor_scrambled, kraus_L, kraus_R, branches_dim_factor=0.5))
#     purification_non_overlapping = kraus_operators_to_purification_non_overlapping(tensor_scrambled,  kraus_L, kraus_R)

#     assert np.allclose(purification, purification_non_overlapping)


# %%
