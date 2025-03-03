"""A minimal change to the code from Miguel Frias Perez in "Converting long-range entanglement into mixture: tensor-network approach to local equilibration https://arxiv.org/abs/2308.04291  " to allow the decomposition's middle tensor to also have a purification leg connected to the left and right degrees of freedom, via a CP decomposition between the fast and slow degrees of freedom, rather than a truncated SVD. Or equivalently, finding block diagonal structure where the blocks are no longer constrained to be identical."""
# %%

import time
from typing import Any, TypeAlias

import numpy as np
import torch
from jaxtyping import Complex
from opt_einops import einsum, rearrange, repeat
from scipy.optimize import minimize

import wavefunction_branching.measure as measure
from wavefunction_branching.decompositions.bell_identical_blocks import mera_like_optimization
from wavefunction_branching.types import (
    BlockDiagTensor,
    LeftEnvironmentTensor,
    LeftSplittingTensor,
    MPSTensor,
    RightEnvironmentTensor,
    RightSplittingTensor,
    UnitarySplittingTensor,
)
from wavefunction_branching.utils.tensors import make_square

PackedVector: TypeAlias = Complex[np.ndarray, "dPacked"]
# ^ where dPacked = 4*(nBranches*dVirt*dSlow) + 2*(dPhys*nBranches*dSlow*dSlow)
# (from flattenning the real and imaginary parts of a LeftSplittingTensor,
#   a BlockDiagTensor, and a RightSplittingTensor into a single vector)


def pack_LSR(L: LeftSplittingTensor, S: BlockDiagTensor, R: RightSplittingTensor) -> PackedVector:
    """Pack the three complex tensors (a left splitter, a reduced "slow" tensor, and a right spltter)
    into a single real vector (reverse of unpack_to_LSR())"""
    L, S, R = [np.real(L), np.imag(L)], [np.real(S), np.imag(S)], [np.real(R), np.imag(R)]  # type: ignore
    L, S, R = np.array(L).reshape(-1), np.array(S).reshape(-1), np.array(R).reshape(-1)
    packed_vector = list(L) + list(S) + list(R)
    packed_vector = np.array(packed_vector)
    return packed_vector


def unpack_to_LSR(
    packed_vector: PackedVector,
    dPurification: int = 4,
    dVirt: int = 2,
    dSlow: int = 2,
    dPhys: int = 4,
) -> tuple[LeftSplittingTensor, BlockDiagTensor, RightSplittingTensor]:
    """Unpack the three complex tensors (a left splitter, a reduced "slow" tensor, and a right spltter)
    from a single real vector (reverse of pack_LSR())"""
    idx1, idx2 = dPurification * dVirt * dSlow, dPhys * dPurification * dSlow * dSlow
    L, S, R = (
        packed_vector[:idx1].reshape(2, -1),
        packed_vector[idx1 : idx1 + idx2].reshape(2, -1),
        packed_vector[idx1 + idx2 :].reshape(2, -1),
    )
    L, S, R = L[0] + 1j * L[1], S[0] + 1j * S[1], R[0] + 1j * R[1]
    L = L.reshape(int(dPurification / 2), dVirt, dSlow)
    S = S.reshape(dPhys, int(dPurification / 2), dSlow, dSlow)
    R = R.reshape(int(dPurification / 2), dSlow, dVirt)
    return L, S, R


def costFun_numeric(
    packed_vector: PackedVector,
    dSlow: int,
    dPhys: int,
    dPurification: int,
    dVirt: int,
    left_env: LeftEnvironmentTensor,
    right_env: RightEnvironmentTensor,
) -> tuple[Any, PackedVector]:
    """Compute the cost function and the gradient for the heuristic optimization"""
    # Construct the three pieces
    L, S, R = unpack_to_LSR(
        packed_vector, dPurification=dPurification, dVirt=dVirt, dSlow=dSlow, dPhys=dPhys
    )

    L = torch.tensor(L, requires_grad=True)
    S = torch.tensor(S, requires_grad=True)
    R = torch.tensor(R, requires_grad=True)

    left_env = torch.tensor(left_env, requires_grad=False)  # type: ignore
    right_env = torch.tensor(right_env, requires_grad=False)  # type: ignore

    tensor = einsum(
        L,
        S,
        R,
        "nBlocks dVirt_L dSlow_L, dPhys nBlocks dSlow_L dSlow_R, nBlocks dSlow_R dVirt_R"
        "-> dPhys dVirt_L dVirt_R nBlocks",
    )

    rhoLeft = einsum(
        tensor,
        tensor.conj(),
        "dPhys dVirt_L dVirt_R nBlocks, dPhys_c dVirt_L_c dVirt_R nBlocks "
        "-> dPhys dVirt_L                  dPhys_c dVirt_L_c               ",
    )
    rhoRight = einsum(
        tensor,
        tensor.conj(),
        "dPhys dVirt_L dVirt_R nBlocks, dPhys_c dVirt_L dVirt_R_c nBlocks"
        "-> dPhys         dVirt_R          dPhys_c         dVirt_R_c      ",
    )

    normRhoLeft = rhoLeft  # /torch.trace( rhoLeft.reshape(dPhys*dVirt, -1))
    normRhoRight = rhoRight  # /torch.trace(rhoRight.reshape(dPhys*dVirt, -1))
    deltaRhoLeft = normRhoLeft - left_env
    deltaRhoRight = normRhoRight - right_env

    wLeft, vLeft = torch.linalg.eigh(deltaRhoLeft.reshape(dPhys * dVirt, -1))
    wRight, vRight = torch.linalg.eigh(deltaRhoRight.reshape(dPhys * dVirt, -1))

    # Define for later returning the cost function
    cost = 0.5 * torch.sum(torch.abs(wLeft) ** 2) + 0.5 * torch.sum(
        torch.abs(wRight) ** 2
    )  # + 0.5*cost_blockdiag - 0.01*norms_prod   #+ torch.abs(1.0-norms_sum)**2

    # Numerically calculate the derivatives
    cost.backward()
    grad_L = L.grad
    grad_S = S.grad
    grad_R = R.grad
    assert grad_L is not None
    assert grad_S is not None
    assert grad_R is not None
    packed_grad = pack_LSR(
        grad_L.detach().cpu().numpy(), grad_S.detach().cpu().numpy(), grad_R.detach().cpu().numpy()
    )
    cost_value = cost.detach().cpu().numpy()
    return (cost_value, packed_grad)


def traceDistance(
    packed_vector: PackedVector,
    dSlow: int,
    dPhys: int,
    dPurification: int,
    dVirt: int,
    left_env: LeftEnvironmentTensor,
    right_env: RightEnvironmentTensor,
) -> list[float]:
    # Construct the three pieces
    L, S, R = unpack_to_LSR(
        packed_vector, dPurification=dPurification, dVirt=dVirt, dSlow=dSlow, dPhys=dPhys
    )

    tensor = np.tensordot(S, L, axes=(1, -1))
    tensor = np.tensordot(tensor, R, axes=(1, 1))
    tensor = np.transpose(tensor, axes=(0, 2, 4, 1, 3)).reshape(dPhys, dVirt, dVirt, dPurification)

    rhoLeft, rhoRight = (
        np.tensordot(tensor, np.conj(tensor), axes=([-2, -1], [-2, -1])),
        np.tensordot(tensor, np.conj(tensor), axes=([-3, -1], [-3, -1])),
    )
    normRhoLeft, normRhoRight = (
        rhoLeft / np.trace(rhoLeft.reshape(dPhys * dVirt, -1)),
        rhoRight / np.trace(rhoRight.reshape(dPhys * dVirt, -1)),
    )
    normRhoLeft, normRhoRight = normRhoLeft - left_env, normRhoRight - right_env

    wLeft, vLeft = np.linalg.eigh(normRhoLeft.reshape(dPhys * dVirt, -1))
    wRight, vRight = np.linalg.eigh(normRhoRight.reshape(dPhys * dVirt, -1))
    return [0.5 * np.sum(np.abs(wLeft)), 0.5 * np.sum(np.abs(wRight))]


def myTraceDistance(
    packed_vector: PackedVector,
    tensor: MPSTensor,
    dSlow: int,
    dPhys: int,
    dPurification: int,
    dVirt: int,
    left_env: LeftEnvironmentTensor,
    right_env: RightEnvironmentTensor,
    initial_guess: None | PackedVector = None,
) -> dict:
    out = {}
    # Construct the three pieces
    L, S, R = unpack_to_LSR(
        packed_vector, dPurification=dPurification, dVirt=dVirt, dSlow=dSlow, dPhys=dPhys
    )

    if initial_guess is not None:
        L0, S0, R0 = unpack_to_LSR(
            initial_guess, dPurification=dPurification, dVirt=dVirt, dSlow=dSlow, dPhys=dPhys
        )
        out["similarity_L0_to_L"] = abs(L0.flatten() @ L.flatten().conj()) / np.sqrt(
            L0.flatten() @ L0.flatten().conj() * L.flatten() @ L.flatten().conj()
        )
        out["similarity_S0_to_S"] = abs(S0.flatten() @ S.flatten().conj()) / np.sqrt(
            S0.flatten() @ S0.flatten().conj() * S.flatten() @ S.flatten().conj()
        )
        out["similarity_R0_to_R"] = abs(R0.flatten() @ R.flatten().conj()) / np.sqrt(
            R0.flatten() @ R0.flatten().conj() * R.flatten() @ R.flatten().conj()
        )
        LSR0 = einsum(
            L0,
            S0,
            R0,
            "nBlocks dVirt_L dSlow_L, dPhys nBlocks dSlow_L dSlow_R, nBlocks dSlow_R dVirt_R"
            "-> dPhys dVirt_L dVirt_R nBlocks",
        )
        LSR = einsum(
            L,
            S,
            R,
            "nBlocks dVirt_L dSlow_L, dPhys nBlocks dSlow_L dSlow_R, nBlocks dSlow_R dVirt_R"
            "-> dPhys dVirt_L dVirt_R nBlocks",
        )
        out["similarity_LSR0_to_LSR"] = abs(LSR0.flatten() @ LSR.flatten().conj()) / np.sqrt(
            LSR0.flatten() @ LSR0.flatten().conj() * LSR.flatten() @ LSR.flatten().conj()
        )

    block_similarity = einsum(S, np.conj(S), "p b l r, p bc l r -> b bc")

    # Construct the three pieces
    L = torch.tensor(L, requires_grad=False)
    S = torch.tensor(S, requires_grad=False)
    R = torch.tensor(R, requires_grad=False)

    left_env = torch.tensor(left_env, requires_grad=False)  # type: ignore
    right_env = torch.tensor(right_env, requires_grad=False)  # type: ignore

    LSR = einsum(
        L,
        S,
        R,
        "nBlocks dVirt_L dSlow_L, dPhys nBlocks dSlow_L dSlow_R, nBlocks dSlow_R dVirt_R"
        "-> dPhys dVirt_L dVirt_R nBlocks",
    )

    rhoLeft = einsum(
        LSR,
        LSR.conj(),
        "dPhys dVirt_L dVirt_R nBlocks, dPhys_c dVirt_L_c dVirt_R nBlocks "
        "-> dPhys dVirt_L                  dPhys_c dVirt_L_c               ",
    )
    rhoRight = einsum(
        LSR,
        LSR.conj(),
        "dPhys dVirt_L dVirt_R nBlocks, dPhys_c dVirt_L dVirt_R_c nBlocks"
        "-> dPhys         dVirt_R          dPhys_c         dVirt_R_c      ",
    )

    normRhoLeft = rhoLeft / torch.trace(rhoLeft.reshape(dPhys * dVirt, -1))
    normRhoRight = rhoRight / torch.trace(rhoRight.reshape(dPhys * dVirt, -1))
    deltaRhoLeft = normRhoLeft - left_env
    deltaRhoRight = normRhoRight - right_env

    wLeft, vLeft = torch.linalg.eigh(deltaRhoLeft.reshape(dPhys * dVirt, -1))
    wRight, vRight = torch.linalg.eigh(deltaRhoRight.reshape(dPhys * dVirt, -1))

    # Check norms
    norms = einsum(
        LSR, LSR.conj(), "dPhys dVirt_L dVirt_R nBlocks,  dPhys dVirt_L dVirt_R nBlocks -> nBlocks"
    )
    norms_sum = torch.sum(norms)
    norms /= norms_sum
    norms_prod = torch.prod(torch.abs(norms))

    # Check for block diagonality (records on the left and right)
    res_L = einsum(
        L,
        np.conj(L),
        "nBlocks dVirt_L dSlow_L, nBlocks_c dVirt_L dSlow_L_c -> nBlocks nBlocks_c dSlow_L dSlow_L_c ",
    )
    res_L.diagonal(dim1=0, dim2=1).zero_()  # type: ignore
    res_R = einsum(
        R,
        np.conj(R),
        "nBlocks dSlow_R dVirt_R, nBlocks_c dSlow_R_c dVirt_R -> nBlocks nBlocks_c dSlow_R dSlow_R_c ",
    )
    res_R.diagonal(dim1=0, dim2=1).zero_()  # type: ignore
    assert isinstance(res_L, torch.Tensor)
    assert isinstance(res_R, torch.Tensor)
    L_block_diagonality_error = torch.sum(torch.abs(res_L) ** 2)
    R_block_diagonality_error = torch.sum(torch.abs(res_R) ** 2)

    # Define for later returning the cost function
    cost_LM_MR = 0.5 * torch.sum(torch.abs(wLeft)) + 0.5 * torch.sum(torch.abs(wRight))

    # Characterize the quality of the decomposition
    out["L_block_diagonality_error"] = L_block_diagonality_error
    out["R_block_diagonality_error"] = R_block_diagonality_error
    out["cost_LM_MR"] = cost_LM_MR
    out["norms_prod"] = norms_prod
    out["norms_sum"] = abs(norms_sum)
    for i in range(len(norms)):
        out[f"norms_{i}"] = norms[i]
    for i in range(block_similarity.shape[0]):
        for j in range(block_similarity.shape[1]):
            if j > i:
                out[f"block_{i}_{j}_similarity"] = abs(block_similarity[i, j]) / np.sqrt(
                    block_similarity[i, i] * block_similarity[j, j]
                )

    theta_purified = einsum(
        L,
        S,
        R,
        "nBlocks dVirt_L dSlow_L, dPhys nBlocks dSlow_L dSlow_R, nBlocks dSlow_R dVirt_R -> nBlocks dPhys dVirt_L dVirt_R",
    )

    density_matrix_orig = einsum(
        tensor, np.conj(tensor), "p  l  r ,           pc l r      ->  p pc"
    )
    density_matrix_L_orig = einsum(
        tensor, np.conj(tensor), "p  l  r ,           pc l rc      ->  p r pc rc"
    )
    density_matrix_R_orig = einsum(
        tensor, np.conj(tensor), "p  l  r ,           pc lc r      ->  p l pc lc"
    )
    for pure in (True, False):
        if pure:
            density_matrix_new = einsum(
                theta_purified,
                np.conj(theta_purified),
                "b p  l  r ,           bc pc l r      ->  p pc",
            )
            density_matrix_L_new = einsum(
                theta_purified,
                np.conj(theta_purified),
                "b p  l  r ,           bc pc l rc     ->  p r pc rc",
            )
            density_matrix_R_new = einsum(
                theta_purified,
                np.conj(theta_purified),
                "b p  l  r ,           bc pc lc r     ->  p l pc lc",
            )
        else:
            density_matrix_new = einsum(
                theta_purified,
                np.conj(theta_purified),
                "b p  l  r ,           b pc l r      ->  p pc",
            )
            density_matrix_L_new = einsum(
                theta_purified,
                np.conj(theta_purified),
                "b p  l  r ,           b pc l rc     ->  p r pc rc",
            )
            density_matrix_R_new = einsum(
                theta_purified,
                np.conj(theta_purified),
                "b p  l  r ,           b pc lc r     ->  p l pc lc",
            )

        for normalize in (True,):
            suffix = "_pure" if pure else "_mixed"
            suffix += "_normalized" if normalize else ""
            suffix += "_" * (30 - len(suffix))
            # Characterize the quality of the decomposition
            trace_distances = measure.LMR_trace_distances(
                tensor, theta_purified.clone().detach().cpu().numpy()
            )
            for key in trace_distances:
                out[key + suffix + "_" * (50 - len(suffix) - len(key))] = trace_distances[key]

    return out


def tensor_u1_u2_to_LSR(
    tensor: MPSTensor,
    u1: UnitarySplittingTensor,
    u2: UnitarySplittingTensor,
) -> tuple[LeftSplittingTensor, BlockDiagTensor, RightSplittingTensor]:
    # Make sure the dimensions match up
    dVirt, dSlow, nBlocks = u1.shape
    assert u2.shape[0] == dVirt
    assert tensor.shape[1] == dVirt
    assert tensor.shape[2] == dVirt
    dPhys, dVirt, dVirt = tensor.shape

    aux = einsum(np.conj(u1), tensor, np.conj(u2), "L l bl,      p L R,  R r br  ->  p l r bl br")
    aux = rearrange(aux, "p l r   bl br -> (p l r) (bl br)")

    # Split the top (slow, physical) from the bottom (fast, virtual)
    tops, vertical_spectrum, bottoms = np.linalg.svd(aux, full_matrices=False)

    # S = tops[:, :nBlocks].reshape(dPhys, nBlocks, dSlow, dSlow)
    S = tops[:, 0].reshape(dPhys, dSlow, dSlow)
    S = repeat(S, "p l r -> p b l r", b=nBlocks)
    fast = bottoms[0].reshape(2, 2)
    U_fast, s_fast, Vh_fast = np.linalg.svd(fast)
    s_fast = s_fast**0.5
    fast_L = U_fast @ np.diag(s_fast)
    fast_R = np.diag(s_fast) @ Vh_fast
    L = einsum(u1, fast_L, "L l bl, bl b -> b L l")
    R = einsum(fast_R, u2, "b br, R r br -> b r R")
    return L, S, R


def heuristic_optimization_LSR(
    tensor: MPSTensor,
    L: LeftSplittingTensor,
    S: BlockDiagTensor,
    R: RightSplittingTensor,
    dPurification: int = 4,
    maxiter: int = 5000,
    numeric_grad=False,
) -> tuple[LeftSplittingTensor, BlockDiagTensor, RightSplittingTensor]:
    """Perform gradient descent optimization to increase the accuracy of the decomposition"""
    startTime = time.time()

    print(f"numeric_grad = {numeric_grad}")

    cost_function = costFun_numeric  # if numeric_grad else costFun

    dPhys, nBlocks, dSlow, dSlow_R = S.shape
    assert dSlow == dSlow_R
    assert dPhys == tensor.shape[0]
    assert dSlow == L.shape[2]
    assert dSlow_R == R.shape[1]
    dPhys, dVirt, dVirt = tensor.shape
    guess = pack_LSR(L, S, R)

    # Pre-compute the left and right environment matrices
    left_env, right_env = (
        np.tensordot(tensor, np.conj(tensor), axes=(-1, -1)),
        np.tensordot(tensor, np.conj(tensor), axes=(1, 1)),
    )

    initial_guess = guess.copy()

    # Wrap the optimization function so that it outputs some information to the screen
    Nfeval = 0

    def callbackF(guess, output_every=50):
        nonlocal Nfeval
        # nonlocal tensor
        # nonlocal dSlow
        # nonlocal dPhys
        # nonlocal dPurification
        # nonlocal dVirt
        # nonlocal left_env
        # nonlocal right_env
        # nonlocal cost_function
        # nonlocal initial_guess
        # if Nfeval % output_every == 0:
        #     mytracedist = myTraceDistance(guess, tensor, dSlow, dPhys, dPurification, dVirt, left_env, right_env, initial_guess=initial_guess)
        #     loss = cost_function(guess, dSlow, dPhys, dPurification, dVirt, left_env, right_env)
        #     print(f'Nfeval = {Nfeval:4d}  costFun = {loss[0]:.2E}, jac_magintude = {np.linalg.norm(loss[1].flatten()):.2E}, jac.shape = {loss[1].shape}, walltime = {time.time() - startTime:.2f}')
        #     for key in sorted(mytracedist.keys()):
        #         print(f'    {key} = {mytracedist[key]:.2E}')
        Nfeval += 1

    # Perform the gradient descent
    result = minimize(
        cost_function,
        guess,
        (dSlow, dPhys, dPurification, dVirt, left_env, right_env),
        method="L-BFGS-B",
        callback=callbackF,
        jac=True,
        tol=1e-15,
        options={"maxiter": maxiter},
    )
    callbackF(result.x, output_every=1)

    L, S, R = unpack_to_LSR(
        result.x, dPurification=dPurification, dVirt=dVirt, dSlow=dSlow, dPhys=dPhys
    )
    return L, S, R


def heuristic_optimization(
    tensor: MPSTensor,
    u1: UnitarySplittingTensor,
    u2: UnitarySplittingTensor,
    dPurification: int = 4,
    maxiter: int = 5000,
    numeric_grad=False,
) -> tuple[LeftSplittingTensor, BlockDiagTensor, RightSplittingTensor]:
    """Perform gradient descent optimization to increase the accuracy of the decomposition"""
    startTime = time.time()

    # Make sure the dimensions match up
    dVirt, dSlow, xFast = u1.shape
    assert u2.shape[0] == dVirt
    assert tensor.shape[1] == dVirt
    assert tensor.shape[2] == dVirt
    dPhys, dVirt, dVirt = tensor.shape

    # Construct the inital guess to pass to the optimization
    L, S, R = tensor_u1_u2_to_LSR(tensor, u1, u2)
    return heuristic_optimization_LSR(
        tensor, L, S, R, dPurification=dPurification, maxiter=maxiter, numeric_grad=numeric_grad
    )


def combined_optimization(
    tensor: MPSTensor,
    tolEntropy: float = 9999.0,  # reasonable: 1e-2
    tolNegativity: float = -9999.0,  # reasonable: 0.2
    n_attempts_iterative: int = 10,
    n_iterations_per_attempt: int = 10**4,
    maxiter_heuristic: int = 5000,
    nBlocks: int = 2,
    dPurification: int = 4,
    early_stopping: bool = True,
    numeric_grad=False,
) -> tuple[LeftSplittingTensor, BlockDiagTensor, RightSplittingTensor, dict]:
    tensor = make_square(tensor, 2)
    dPhys, dVirt, dVirt_R = tensor.shape
    assert dVirt == dVirt_R
    dSlow = int(dVirt / nBlocks)

    print("MERA-like optimization:")
    u1, tensor, u2, info = mera_like_optimization(
        tensor,
        nBlocks,
        tolNegativity=tolNegativity,
        n_attempts=n_attempts_iterative,
        n_iterations_per_attempt=n_iterations_per_attempt,
        early_stopping=early_stopping,
    )

    info["rejected"] = False
    if info["entropy"] > tolEntropy or info["negativity"] < tolNegativity:
        info["rejected"] = True
        print(
            "    Further optimization was rejected as initial mera-like optimization failed to find a good decomposition."
        )
        print(f"    entropy = {info['entropy']} (tolEntropy = {tolEntropy})")
        print(f"    negativity = {info['negativity']} (tolNegativity = {tolNegativity})")

    # If the decomposition found is below the threshold and mediates long range entanglement,
    # carry out the heuristic optimization
    if maxiter_heuristic < 1 or info["rejected"]:
        L, S, R = tensor_u1_u2_to_LSR(tensor, u1, u2)
        return L, S, R, info
    else:
        print("    Heuristic optimization:")
        L, S, R = heuristic_optimization(
            tensor,
            u1,
            u2,
            dPurification=dPurification,
            maxiter=maxiter_heuristic,
            numeric_grad=numeric_grad,
        )
        return L, S, R, info
