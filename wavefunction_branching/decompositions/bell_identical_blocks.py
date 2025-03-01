"""Code adapted from Miguel Frias Perez: Converting long-range entanglement into mixture: tensor-network approach to local equilibration https://arxiv.org/abs/2308.04291"""

import time
from typing import TypeAlias

import numpy as np
from jaxtyping import Complex
from opt_einops import einsum, rearrange
from scipy.optimize import minimize

import wavefunction_branching.measure as measure
from wavefunction_branching.types import (
    FastVector,
    LeftEnvironmentTensor,
    LeftSplittingTensor,
    MPSTensor,
    RightEnvironmentTensor,
    RightSplittingTensor,
    SlowTensor,
    UnitarySplittingTensor,
)
from wavefunction_branching.utils.tensors import make_square, unitize

PackedVector: TypeAlias = Complex[np.ndarray, "dPacked"]
# ^ where dPacked = 4*(nBranches*dVirt*dSlow) + 2*(dPhys*dSlow*dSlow)
# (from flattenning the real and imaginary parts of a LeftSplittingTensor,
#  a SlowTensor, and a RightSplittingTensor into a single vector)


def update_left_unitary(
    tensor: MPSTensor, top: SlowTensor, bottom: FastVector, u2: UnitarySplittingTensor
) -> UnitarySplittingTensor:
    """Optimize over the left unitary by computing and unitizing its environment (classic MERA / Vidal iterative unitary optimization)"""
    xSlow = top.shape[1]
    xFast = bottom.shape[0]
    env = einsum(
        top, u2, bottom, np.conj(tensor), "p l r,  R r b,  b,        p L R      ->     L l b"
    )
    env = rearrange(env, "L l b -> (l b) L")
    u1 = np.transpose(np.conj(unitize(env)))
    u1 = rearrange(u1, "L (l b) -> L l b", l=xSlow, b=xFast)
    return u1


def update_right_unitary(
    tensor: MPSTensor, top: SlowTensor, bottom: FastVector, u1: UnitarySplittingTensor
) -> UnitarySplittingTensor:
    """Optimize over the right unitary by computing and unitizing its environment (classic MERA / Vidal iterative unitary optimization)"""
    xSlow = top.shape[2]
    xFast = bottom.shape[0]
    env = einsum(
        top, u1, bottom, np.conj(tensor), "p l r,  L l b,  b,        p L R      ->     R r b"
    )
    env = rearrange(env, "R r b -> (r b) R")
    u2 = np.transpose(np.conj(unitize(env)))
    u2 = rearrange(u2, "R (r b) -> R r b", r=xSlow, b=xFast)
    return u2


def update_middle(
    tensor: MPSTensor, u1: UnitarySplittingTensor, u2: UnitarySplittingTensor
) -> tuple[UnitarySplittingTensor, SlowTensor, FastVector, UnitarySplittingTensor]:
    xFast = u1.shape[-1]
    # Update the bottom by finding the dominant eigenvector of rho_fast (the bottom's environment)
    overlap = einsum(
        np.conj(u1), tensor, np.conj(u2), "L l bl,     p L R,  R r br  ->  bl br p l r"
    )
    rho_fast = einsum(overlap, np.conj(overlap), "bl br p l r,  blc brc p l r   ->  bl br blc brc")
    rho_fast = rho_fast.reshape(xFast**2, xFast**2)
    w, v = np.linalg.eigh(rho_fast)  # The eigenvectors act on the conjugate legs
    bottom = v[:, -1].reshape(xFast, xFast)  # (normalized)

    # Update the top by contracting the bottom into np.conj(u1) @ tensor @ np.conj(u2)
    top = einsum(overlap, bottom, "bl br p l r,  bl br  ->  p l r")

    # Update u1 and u2 by taking the SVD of the bottom
    u, bottom, v = np.linalg.svd(bottom)
    u1, u2 = np.tensordot(u1, u, axes=(-1, 0)), np.tensordot(u2, v, axes=(-1, -1))
    return u1, u2, top, bottom


def approxTensor(
    top: SlowTensor, bottom: FastVector, u1: UnitarySplittingTensor, u2: UnitarySplittingTensor
) -> MPSTensor:
    # Take the tensors top bottom u1 and u2 and construct and approximation to the time evolved tensor
    appTensor = np.tensordot(top, u1, axes=(1, 1))
    appTensor = np.tensordot(appTensor, np.diag(bottom), axes=(-1, 0))
    appTensor = np.tensordot(appTensor, u2, axes=([1, 3], [1, 2]))
    return appTensor


def distance(
    tensor: MPSTensor,
    top: SlowTensor,
    bottom: FastVector,
    u1: UnitarySplittingTensor,
    u2: UnitarySplittingTensor,
) -> float:
    # Compute the distance between the actual tensor from the canonical center and the approximation
    A = tensor.flatten()
    B = approxTensor(top, bottom, u1, u2).flatten()
    distance = np.sum(abs(A - B) ** 2)
    assert isinstance(distance, float)
    return distance


def factorizeBlock(
    tensor: MPSTensor, d: int, xSlow: int, xFast: int, n_steps=10**4, early_stopping=True, **kwargs
) -> tuple[UnitarySplittingTensor, UnitarySplittingTensor, SlowTensor, FastVector]:
    """Mera-like optimization to factorize the tensor"""

    # TODO: FINISH ADDING A DECENT INITIALIZATION WITH BSVD
    # u1, B, u2, block_sizes = bsvd.block_svd(tensor, **kwargs)
    # B_purified = bsvd.blockdiag_to_purification(u1, B, u2, block_sizes)
    # B_purified = B_purified[..., :xSlow, :xSlow]
    # b, p, l, r = B_purified.shape
    # rearrange(B_purified, 'b p l r -> b (p l r)')

    # Initialize a set of random tensors
    top, bottom = np.random.rand(d, xSlow, xSlow), np.ones(xFast) / np.sqrt(xFast)
    u1, u2 = (
        np.random.rand(xSlow * xFast, xSlow * xFast),
        np.random.rand(xSlow * xFast, xSlow * xFast),
    )
    u1, u2 = (
        np.linalg.svd(u1)[0].reshape(xSlow * xFast, xSlow, xFast),
        np.linalg.svd(u2)[0].reshape(xSlow * xFast, xSlow, xFast),
    )

    norm = approxTensor(top, bottom, u1, u2).flatten()
    norm = norm @ np.conj(norm)
    top = top / np.sqrt(norm)
    distOld = distance(tensor, top, bottom, u1, u2)

    for step in range(n_steps):
        # Optimize over the left unitary. Compute the environment
        u1 = update_left_unitary(tensor, top, bottom, u2)

        # Optimize over the right unitary. Compute the environment
        u2 = update_right_unitary(tensor, top, bottom, u1)

        # Optimize over the top tensor and lower tensor. Compute the environment
        u1, u2, top, bottom = update_middle(tensor, u1, u2)

        dist = distance(tensor, top, bottom, u1, u2)

        if early_stopping and (
            abs(dist - distOld) / distOld < 1e-7
            or dist < 1e-10
            or 2 * np.log2(np.sum(bottom)) < 0.1
        ):
            return u1, u2, top, bottom
        else:
            distOld = dist
    return u1, u2, top, bottom


def mera_like_optimization(
    tensor: MPSTensor,
    xFast: int = 2,
    tolNegativity: float = 0.2,
    n_attempts: int = 10,
    n_iterations_per_attempt: int = 10**4,
    early_stopping: bool = True,
) -> tuple[UnitarySplittingTensor, MPSTensor, UnitarySplittingTensor, dict]:
    unitaries, entropies, negativities, attempts = [], [], [], []

    # Run 10 times the mera-like optimization, and select the best decomposition
    for attempt in range(n_attempts):
        print("   attempt", attempt)
        shape = list(tensor.shape)
        tensor = tensor.reshape(-1, shape[-2], shape[-1])

        norm = tensor.flatten() @ np.conj(tensor.flatten())
        tensor = tensor / np.sqrt(norm)

        shape = list(tensor.shape)
        u1, u2, top, bottom = factorizeBlock(
            tensor,
            shape[0],
            int(shape[1] / xFast),
            xFast,
            n_steps=n_iterations_per_attempt,
            early_stopping=early_stopping,
        )

        # Compute the entanglement between slow and fast degrees of freedom
        overlap = einsum(
            np.conj(u1), tensor, np.conj(u2), "L l bl,     p L R,  R r br  ->  bl br p l r"
        )
        rho_fast = einsum(
            overlap, np.conj(overlap), "bl br p l r,  blc brc p l r   ->  bl br blc brc"
        )
        w, v = np.linalg.eigh(rearrange(rho_fast, "bl br blc brc -> (bl br) (blc brc)"))
        w = w / np.sum(w)
        sch = np.copy(w)
        v = v[:, -1].reshape(xFast, xFast)

        # Compute the logarithmic negativity (the entanglement from L to R)
        w = np.linalg.eigh(rearrange(rho_fast, "bl br blc brc -> (bl blc) (br brc)"))[0]
        w = w / np.sum(w)

        overlaps = einsum(
            u1,
            top,
            bottom,
            u2,
            np.conj(u1),
            np.conj(top),
            np.conj(bottom),
            np.conj(u2),
            "L l b,  p l r,  b,       R r b, L lc bc,      p lc rc,       bc,               R rc bc  ->  b bc",
        )  # type: ignore
        # print('        ee =', -np.sum(np.log2(sch)*sch), ', logNeg =', np.log2(np.sum(abs(w))), 2*np.log2(np.sum(np.linalg.svd(v)[1])))
        # print(f'        norms = {np.diag(overlaps)}, overlaps = {[overlaps[0,1], overlaps[1,0]]}')
        # print(f'        vertical spectrum = {np.sqrt(sch[::-1])}, sum**2 = {np.sum(sch[:-1])}, sum = {np.sum(sch)}')
        # print('        ----------------------------------------------------')

        attempts.append(attempt)
        entropies.append(-np.sum(np.log2(sch) * sch))
        negativities.append(np.log2(np.sum(abs(w))))
        unitaries.append([np.copy(u1), np.copy(u2)])

    entropies = np.array(entropies)
    negativities = np.array(negativities)
    attempts = np.array(attempts)

    # Try to find an attempt with a decent tolNegativity
    valid_attempts = attempts[negativities > tolNegativity]
    if len(valid_attempts) == 0:
        valid_attempts = attempts

    # Within that, find the attempt with the lowest slow/fast entanglement entropy
    idx = np.argmin(entropies[valid_attempts])
    idx = valid_attempts[idx]

    info = {"entropy": entropies[idx], "negativity": negativities[idx], "idx": idx}
    return unitaries[idx][0], tensor, unitaries[idx][1], info


def pack_LSR(L: LeftSplittingTensor, S: SlowTensor, R: RightSplittingTensor) -> PackedVector:
    """Pack the three complex tensors (a left splitter, a reduced "slow" tensor, and a right spltter)
    into a single real vector (reverse of unpack_to_LSR())"""
    L, S, R = [np.real(L), np.imag(L)], [np.real(S), np.imag(S)], [np.real(R), np.imag(R)]  # type: ignore
    L, S, R = np.array(L).reshape(-1), np.array(S).reshape(-1), np.array(R).reshape(-1)
    packed_vector = list(L) + list(S) + list(R)
    packed_vector = np.array(packed_vector)
    return packed_vector


def unpack_to_LSR(
    packed_vector: PackedVector,
    dPurification: int = 1,
    dVirt: int = 2,
    dSlow: int = 2,
    dPhys: int = 4,
) -> tuple[LeftSplittingTensor, SlowTensor, RightSplittingTensor]:
    """Unpack the three complex tensors (a left splitter, a reduced "slow" tensor, and a right spltter)
    from a single real vector (reverse of pack_LSR())"""
    idx1, idx2 = dPurification * dVirt * dSlow, 2 * dPhys * dSlow * dSlow

    L, S, R = (
        packed_vector[:idx1].reshape(2, -1),
        packed_vector[idx1 : idx1 + idx2].reshape(2, -1),
        packed_vector[idx1 + idx2 :].reshape(2, -1),
    )
    L, S, R = L[0] + 1j * L[1], S[0] + 1j * S[1], R[0] + 1j * R[1]

    L = L.reshape(int(dPurification / 2), dVirt, dSlow)
    S = S.reshape(dPhys, dSlow, dSlow)
    R = R.reshape(int(dPurification / 2), dSlow, dVirt)
    return L, S, R


def costFun_discardclassical(
    packed_vector: PackedVector,
    dSlow: int,
    dPhys: int,
    dPurification: int,
    dVirt: int,
    left_env: LeftEnvironmentTensor,
    right_env: RightEnvironmentTensor,
) -> tuple[float, PackedVector]:
    """Compute the cost function and the gradient for the heuristic optimization"""
    # Construct the three pieces
    L, S, R = unpack_to_LSR(
        packed_vector, dPurification=dPurification, dVirt=dVirt, dSlow=dSlow, dPhys=dPhys
    )

    # Re-construct the approximation to the original central tensor (with purification legs)
    tensor = einsum(
        L,
        S,
        R,
        "xFast_L dVirt_L dSlow_L,  dPhys dSlow_L dSlow_R,  xFast_R dSlow_R dVirt_R"
        "-> dPhys dVirt_L dVirt_R xFast_L xFast_R",
    )

    # Group the fast legs into a single purification leg
    tensor = rearrange(
        tensor, "dPhys dVirt_L dVirt_R  xFast_L xFast_R -> dPhys dVirt_L dVirt_R (xFast_L xFast_R)"
    )

    # Trace over the right bond dimension (and purification legs) to create rhoLeft
    rhoLeft = einsum(
        tensor,
        np.conj(tensor),
        "dPhys dVirt_L dVirt_R xFast2,  dPhys_c dVirt_L_c dVirt_R xFast2 "
        "-> dPhys dVirt_L                  dPhys_c dVirt_L_c               ",
    )

    # Trace over the left bond dimension (and purification legs) to create rhoRight
    rhoRight = einsum(
        tensor,
        np.conj(tensor),
        "dPhys dVirt_L dVirt_R xFast2,  dPhys_c dVirt_L dVirt_R_c xFast2"
        "-> dPhys         dVirt_R          dPhys_c         dVirt_R_c      ",
    )

    # Normalize the left and right density matrices
    normRhoLeft = rhoLeft / np.trace(rhoLeft.reshape(dPhys * dVirt, -1))
    normRhoRight = rhoRight / np.trace(rhoRight.reshape(dPhys * dVirt, -1))

    # Find the difference in the density matrices from the original left_env and right_env
    # for calculating the trace norm
    deltaRhoLeft = normRhoLeft - left_env
    deltaRhoRight = normRhoRight - right_env
    wLeft, vLeft = np.linalg.eigh(deltaRhoLeft.reshape(dPhys * dVirt, -1))
    wRight, vRight = np.linalg.eigh(deltaRhoRight.reshape(dPhys * dVirt, -1))

    # Define the cost function  as the mean trace distance to the left_env and right_env
    cost = 0.5 * np.sum(wLeft**2) + 0.5 * np.sum(wRight**2)

    # -------------------------------------------------------------------------------------------
    # Compute the gradient of the cost function analytically
    gradLeft = np.conj(vLeft) @ np.diag(wLeft) @ np.transpose(vLeft)
    gradRight = np.conj(vRight) @ np.diag(wRight) @ np.transpose(vRight)
    gradLeft = gradLeft.reshape(dPhys, dVirt, dPhys, dVirt)
    gradRight = gradRight.reshape(dPhys, dVirt, dPhys, dVirt)
    tr_rhoLeft: float = np.trace(rhoLeft.reshape(dPhys * dVirt, -1))
    tr_rhoRight: float = np.trace(rhoRight.reshape(dPhys * dVirt, -1))
    normLeft = gradLeft.reshape(-1) @ rhoLeft.reshape(-1) / tr_rhoLeft**2
    normRight = gradRight.reshape(-1) @ rhoRight.reshape(-1) / tr_rhoRight**2
    deltaRhoLeft_tensor = einsum(
        gradLeft,
        tensor,
        "dPhys dVirt_L dPhys_c dVirt_L_c,  dPhys dVirt_L dVirt_R xFast2 "
        "-> dPhys_c dVirt_L_c dVirt_R xFast2",
    )
    deltaRhoRight_tensor = einsum(
        gradRight,
        tensor,
        "dPhys dVirt_R dPhys_c dVirt_R_c,  dPhys dVirt_L dVirt_R xFast2 "
        "-> dPhys_c dVirt_L dVirt_R_c xFast2",
    )
    gradLeft = -tensor * normLeft + deltaRhoLeft_tensor / tr_rhoLeft
    gradRight = -tensor * normRight + deltaRhoRight_tensor / tr_rhoRight

    grad = 2 * (gradLeft + gradRight)
    grad = grad.reshape(dPhys, dVirt, dVirt, int(dPurification / 2), int(dPurification / 2))

    gradL = einsum(
        grad,
        np.conj(S),
        np.conj(R),
        "dPhys dVirt_L dVirt_R xFast_L xFast_R,  dPhys dSlow_L dSlow_R,  xFast_R dSlow_R dVirt_R"
        "-> xFast_L dVirt_L dSlow_L",
    )

    gradS = einsum(
        grad,
        np.conj(L),
        np.conj(R),
        "dPhys dVirt_L dVirt_R xFast_L xFast_R,  xFast_L dVirt_L dSlow_L,  xFast_R dSlow_R dVirt_R"
        "-> dPhys dSlow_L dSlow_R",
    )

    gradR = einsum(
        grad,
        np.conj(S),
        np.conj(L),
        "dPhys dVirt_L dVirt_R xFast_L xFast_R,  dPhys dSlow_L dSlow_R,  xFast_L dVirt_L dSlow_L"
        "-> xFast_R dSlow_R dVirt_R",
    )

    packed_grad = pack_LSR(gradL, gradS, gradR)
    assert isinstance(cost, float)
    return (cost, packed_grad)


def costFun_keepclassical(
    packed_vector: PackedVector,
    dSlow: int,
    dPhys: int,
    dPurification: int,
    dVirt: int,
    left_env: LeftEnvironmentTensor,
    right_env: RightEnvironmentTensor,
) -> tuple[float, PackedVector]:
    """Compute the cost function and the gradient for the heuristic optimization"""
    # Construct the three pieces
    L, S, R = unpack_to_LSR(
        packed_vector, dPurification=dPurification, dVirt=dVirt, dSlow=dSlow, dPhys=dPhys
    )

    # Re-construct the approximation to the original central tensor (with purification leg)
    tensor = einsum(
        L,
        S,
        R,
        "xFast dVirt_L dSlow_L, dPhys dSlow_L dSlow_R, xFast dSlow_R dVirt_R"
        "-> dPhys dVirt_L dVirt_R xFast",
    )

    # Trace over the right bond dimension (and purification legs) to create rhoLeft
    rhoLeft = einsum(
        tensor,
        np.conj(tensor),
        "dPhys dVirt_L dVirt_R xFast, dPhys_c dVirt_L_c dVirt_R xFast "
        "-> dPhys dVirt_L                dPhys_c dVirt_L_c               ",
    )

    # Trace over the left bond dimension (and purification legs) to create rhoRight
    rhoRight = einsum(
        tensor,
        np.conj(tensor),
        "dPhys dVirt_L dVirt_R xFast, dPhys_c dVirt_L dVirt_R_c xFast"
        "-> dPhys         dVirt_R        dPhys_c         dVirt_R_c      ",
    )

    # Normalize the left and right density matrices
    normRhoLeft = rhoLeft / np.trace(rhoLeft.reshape(dPhys * dVirt, -1))
    normRhoRight = rhoRight / np.trace(rhoRight.reshape(dPhys * dVirt, -1))

    # Find the difference in the density matrices from the original left_env and right_env
    # for calculating the trace norm
    deltaRhoLeft = normRhoLeft - left_env
    deltaRhoRight = normRhoRight - right_env
    wLeft, vLeft = np.linalg.eigh(deltaRhoLeft.reshape(dPhys * dVirt, -1))
    wRight, vRight = np.linalg.eigh(deltaRhoRight.reshape(dPhys * dVirt, -1))

    # Define the cost function  as the mean trace distance to the left_env and right_env
    cost = 0.5 * np.sum(wLeft**2) + 0.5 * np.sum(wRight**2)

    # -------------------------------------------------------------------------------------------
    # Compute the gradient of the cost function analytically
    gradLeft = np.conj(vLeft) @ np.diag(wLeft) @ np.transpose(vLeft)
    gradRight = np.conj(vRight) @ np.diag(wRight) @ np.transpose(vRight)
    gradLeft = gradLeft.reshape(dPhys, dVirt, dPhys, dVirt)
    gradRight = gradRight.reshape(dPhys, dVirt, dPhys, dVirt)
    tr_rhoLeft: float = np.trace(rhoLeft.reshape(dPhys * dVirt, -1))
    tr_rhoRight: float = np.trace(rhoRight.reshape(dPhys * dVirt, -1))
    normLeft = gradLeft.reshape(-1) @ rhoLeft.reshape(-1) / tr_rhoLeft**2
    normRight = gradRight.reshape(-1) @ rhoRight.reshape(-1) / tr_rhoRight**2
    deltaRhoLeft_tensor = einsum(
        gradLeft,
        tensor,
        "dPhys dVirt_L dPhys_c dVirt_L_c,  dPhys dVirt_L dVirt_R xFast "
        "-> dPhys_c dVirt_L_c dVirt_R xFast",
    )
    deltaRhoRight_tensor = einsum(
        gradRight,
        tensor,
        "dPhys dVirt_R dPhys_c dVirt_R_c,  dPhys dVirt_L dVirt_R xFast "
        "-> dPhys_c dVirt_L dVirt_R_c xFast",
    )
    gradLeft = -tensor * normLeft + deltaRhoLeft_tensor / tr_rhoLeft
    gradRight = -tensor * normRight + deltaRhoRight_tensor / tr_rhoRight

    grad = 2 * (gradLeft + gradRight)

    gradL = einsum(
        grad,
        np.conj(S),
        np.conj(R),
        "dPhys dVirt_L dVirt_R xFast,  dPhys dSlow_L dSlow_R,  xFast dSlow_R dVirt_R"
        "-> xFast dVirt_L dSlow_L",
    )

    gradS = einsum(
        grad,
        np.conj(L),
        np.conj(R),
        "dPhys dVirt_L dVirt_R xFast,  xFast dVirt_L dSlow_L,  xFast dSlow_R dVirt_R"
        "-> dPhys dSlow_L dSlow_R",
    )

    gradR = einsum(
        grad,
        np.conj(S),
        np.conj(L),
        "dPhys dVirt_L dVirt_R xFast,  dPhys dSlow_L dSlow_R,  xFast dVirt_L dSlow_L"
        "-> xFast dSlow_R dVirt_R",
    )

    packed_grad = pack_LSR(gradL, gradS, gradR)

    assert isinstance(cost, float)
    return (cost, packed_grad)


# def costFun_keepclassical_numeric(
#         packed_vector: PackedVector,
#         dSlow: int,
#         dPhys: int,
#         dPurification: int,
#         dVirt: int,
#         left_env: LeftEnvironmentTensor,
#         right_env: RightEnvironmentTensor
#         ) -> Tuple[float, PackedVector]:
#     """Compute the cost function and the gradient for the heuristic optimization"""
#     # Construct the three pieces
#     L, S, R = unpack_to_LSR(packed_vector, dPurification=dPurification, dVirt=dVirt, dSlow=dSlow, dPhys=dPhys)
#     L = torch.tensor(L, requires_grad=True)
#     S = torch.tensor(S, requires_grad=True)
#     R = torch.tensor(R, requires_grad=True)

#     left_env = torch.tensor(left_env, requires_grad=False)
#     right_env = torch.tensor(right_env, requires_grad=False)

#     tensor = einsum(    L,                     S,                     R,
#                        'xFast dVirt_L dSlow_L, dPhys dSlow_L dSlow_R, xFast dSlow_R dVirt_R'
#                     '-> dPhys dVirt_L dVirt_R xFast')

#     rhoLeft = einsum(tensor,                          tensor.conj(),
#                         'dPhys dVirt_L dVirt_R xFast, dPhys_c dVirt_L_c dVirt_R xFast '
#                      '-> dPhys dVirt_L                dPhys_c dVirt_L_c               ')
#     rhoRight = einsum(tensor,                          tensor.conj(),
#                          'dPhys dVirt_L dVirt_R xFast, dPhys_c dVirt_L dVirt_R_c xFast'
#                       '-> dPhys         dVirt_R        dPhys_c         dVirt_R_c      ')

#     normRhoLeft  = rhoLeft/torch.trace( rhoLeft.reshape(dPhys*dVirt, -1))
#     normRhoRight = rhoRight/torch.trace(rhoRight.reshape(dPhys*dVirt, -1))
#     deltaRhoLeft  = normRhoLeft  - left_env
#     deltaRhoRight = normRhoRight - right_env

#     wLeft,  vLeft  = torch.linalg.eigh( deltaRhoLeft.reshape(dPhys*dVirt, -1))
#     wRight, vRight = torch.linalg.eigh(deltaRhoRight.reshape(dPhys*dVirt, -1))

#     # Define for later returning the cost function
#     cost = 0.5*torch.sum(wLeft**2) + 0.5*torch.sum(wRight**2)

#     # Numerically calculate the derivatives
#     cost.backward()
#     packed_grad = pack_LSR(
#         L.grad.detach().cpu().numpy(),
#         S.grad.detach().cpu().numpy(),
#         R.grad.detach().cpu().numpy())
#     return (cost.detach().cpu().numpy(), packed_grad)


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
    assert isinstance(wLeft, np.ndarray)
    assert isinstance(wRight, np.ndarray)
    return [0.5 * np.sum(abs(wLeft)), 0.5 * np.sum(abs(wRight))]


def myTraceDistance(
    packed_vector: PackedVector,
    tensor: MPSTensor,
    dSlow: int,
    dPhys: int,
    dPurification: int,
    dVirt: int,
    left_env: LeftEnvironmentTensor,
    right_env: RightEnvironmentTensor,
    keep_classical_correlations=False,
) -> dict:
    out = {}
    # Construct the three pieces
    L, S, R = unpack_to_LSR(
        packed_vector, dPurification=dPurification, dVirt=dVirt, dSlow=dSlow, dPhys=dPhys
    )

    LSR = einsum(
        L,
        S,
        R,
        "nBlocks dVirt_L dSlow_L, dPhys dSlow_L dSlow_R, nBlocks dSlow_R dVirt_R"
        "-> dPhys dVirt_L dVirt_R nBlocks",
    )
    # Check norms
    norms = einsum(
        LSR, LSR.conj(), "dPhys dVirt_L dVirt_R nBlocks,  dPhys dVirt_L dVirt_R nBlocks -> nBlocks"
    )
    norms_sum = np.sum(norms)
    norms /= norms_sum
    norms_prod = np.prod(np.abs(norms))

    # Check for block diagonality (records on the left and right)
    res_L = einsum(
        L,
        np.conj(L),
        "nBlocks dVirt_L dSlow_L, nBlocks_c dVirt_L dSlow_L_c -> nBlocks nBlocks_c dSlow_L dSlow_L_c ",
    )
    for i in range(res_L.shape[0]):
        res_L[i, i] *= 0.0
    res_R = einsum(
        R,
        np.conj(R),
        "nBlocks dSlow_R dVirt_R, nBlocks_c dSlow_R_c dVirt_R -> nBlocks nBlocks_c dSlow_R dSlow_R_c ",
    )
    for i in range(res_R.shape[0]):
        res_R[i, i] *= 0.0
    L_block_diagonality_error = np.sum(np.abs(res_L) ** 2)
    R_block_diagonality_error = np.sum(np.abs(res_R) ** 2)

    out["L_block_diagonality_error"] = L_block_diagonality_error
    out["R_block_diagonality_error"] = R_block_diagonality_error
    out["norms_prod"] = norms_prod
    out["norms_sum"] = abs(norms_sum)

    theta_purifieds = {}
    if keep_classical_correlations:
        theta_purifieds["with_classical_correlations"] = einsum(
            L,
            S,
            R,
            "xFast dVirt_L dSlow_L, dPhys dSlow_L dSlow_R, xFast dSlow_R dVirt_R -> xFast dPhys dVirt_L dVirt_R",
        )
    else:
        theta_purified_no_classical_correlations = einsum(
            L,
            S,
            R,
            "xFast_L dVirt_L dSlow_L, dPhys dSlow_L dSlow_R, xFast_R dSlow_R dVirt_R -> xFast_L xFast_R dPhys dVirt_L dVirt_R",
        )
        theta_purifieds["no_classical_correlations"] = rearrange(
            theta_purified_no_classical_correlations,
            "xFast_L xFast_R dPhys dVirt_L dVirt_R -> (xFast_L xFast_R) dPhys dVirt_L dVirt_R",
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
    for theta_purified_name, theta_purified in theta_purifieds.items():
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
                suffix += "_" + theta_purified_name
                suffix += "_normalized" if normalize else ""
                suffix += "_" * (50 - len(suffix))
                for i in range(len(norms)):
                    out[f"norms_{i}"] = norms[i]
                # Characterize the quality of the decomposition
                trace_distances = measure.LMR_trace_distances(tensor, theta_purified)
                for key in trace_distances:
                    out[key + suffix + "_" * (50 - len(suffix) - len(key))] = trace_distances[key]
    return out


def tensor_u1_u2_to_LSR(
    tensor: MPSTensor,
    u1: UnitarySplittingTensor,
    u2: UnitarySplittingTensor,
    keep_classical_correlations=True,
) -> tuple[LeftSplittingTensor, SlowTensor, RightSplittingTensor]:
    # Make sure the dimensions match up
    dVirt, dSlow, xFast = u1.shape
    assert u2.shape[0] == dVirt
    assert tensor.shape[1] == dVirt
    assert tensor.shape[2] == dVirt
    dPhys, dVirt, dVirt = tensor.shape

    aux = einsum(np.conj(u1), tensor, np.conj(u2), "L l bl,      p L R,  R r br  ->  p l r bl br")
    aux = rearrange(aux, "p l r   bl br -> (p l r) (bl br)")

    # Split the top (slow, physical) from the bottom (fast, virtual)
    tops, vertical_spectrum, bottoms = np.linalg.svd(aux, full_matrices=False)

    S = tops[:, 0].reshape(dPhys, dSlow, dSlow)
    fast = bottoms[0].reshape(2, 2)
    U_fast, s_fast, Vh_fast = np.linalg.svd(fast)
    if keep_classical_correlations:
        s_fast = s_fast**0.5
    fast_L = U_fast @ np.diag(s_fast)
    fast_R = np.diag(s_fast) @ Vh_fast
    L = einsum(u1, fast_L, "L l bl, bl b -> b L l")
    R = einsum(fast_R, u2, "b br, R r br -> b r R")
    return L, S, R


def heuristic_optimization_LSR(
    tensor: MPSTensor,
    L: LeftSplittingTensor,
    S: SlowTensor,
    R: RightSplittingTensor,
    dPurification: int = 4,
    maxiter: int = 5000,
    keep_classical_correlations=True,
) -> tuple[LeftSplittingTensor, SlowTensor, RightSplittingTensor]:
    """Perform gradient descent optimization to increase the accuracy of the decomposition"""
    startTime = time.time()

    costFun = costFun_keepclassical if keep_classical_correlations else costFun_discardclassical

    dPhys, dSlow, dSlow_R = S.shape
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

    # Wrap the optimization function so that it outputs some information to the screen
    Nfeval = 0

    def callbackF(guess, output_every=100):
        nonlocal Nfeval
        # nonlocal tensor
        # nonlocal dSlow
        # nonlocal dPhys
        # nonlocal dPurification
        # nonlocal dVirt
        # nonlocal left_env
        # nonlocal right_env
        # nonlocal costFun
        # if Nfeval % output_every == 0:
        # mytracedist = myTraceDistance(guess, tensor, dSlow, dPhys, dPurification, dVirt, left_env, right_env, keep_classical_correlations=keep_classical_correlations)
        # costfun = costFun(guess, dSlow, dPhys, dPurification, dVirt, left_env, right_env)
        # print(f'Nfeval = {Nfeval:4d}  costFun = {costfun[0]:.2E}, jac_magintude = {np.linalg.norm(costfun[1].flatten()):.2E}, jac.shape = {costfun[1].shape}, walltime = {time.time() - startTime:.2f}')
        # for key in sorted(mytracedist.keys()):
        #     print(f'    {key} = {mytracedist[key]:.2E}')
        Nfeval += 1

    # Perform the gradient descent
    result = minimize(
        costFun,
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
    keep_classical_correlations=True,
) -> tuple[LeftSplittingTensor, SlowTensor, RightSplittingTensor]:
    """Perform gradient descent optimization to increase the accuracy of the decomposition"""
    startTime = time.time()

    # Make sure the dimensions match up
    dVirt, dSlow, xFast = u1.shape
    assert u2.shape[0] == dVirt
    assert tensor.shape[1] == dVirt
    assert tensor.shape[2] == dVirt
    dPhys, dVirt, dVirt = tensor.shape

    # Construct the inital guess to pass to the optimization
    L, S, R = tensor_u1_u2_to_LSR(
        tensor, u1, u2, keep_classical_correlations=keep_classical_correlations
    )
    return heuristic_optimization_LSR(
        tensor,
        L,
        S,
        R,
        dPurification=dPurification,
        maxiter=maxiter,
        keep_classical_correlations=keep_classical_correlations,
    )


def combined_optimization(
    tensor: MPSTensor,
    tolEntropy: float = 9999.0,  # reasonable: 1e-2
    tolNegativity: float = -9999.0,  # reasonable: 0.2
    n_attempts_iterative: int = 10,
    n_iterations_per_attempt: int = 10**4,
    maxiter_heuristic: int = 5000,
    xFast: int = 2,
    dPurification: int = 4,
    early_stopping: bool = True,
    keep_classical_correlations: bool = True,
) -> tuple[LeftSplittingTensor, SlowTensor, RightSplittingTensor, dict]:
    tensor = make_square(tensor, 2)
    dPhys, dVirt, dVirt_R = tensor.shape
    assert dVirt == dVirt_R
    dSlow = int(dVirt / xFast)

    print("MERA-like optimization:")
    u1, tensor, u2, info = mera_like_optimization(
        tensor,
        xFast=xFast,
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
        L, S, R = tensor_u1_u2_to_LSR(
            tensor, u1, u2, keep_classical_correlations=keep_classical_correlations
        )
        return L, S, R, info
    else:
        print("    Heuristic optimization:")
        L, S, R = heuristic_optimization(
            tensor,
            u1,
            u2,
            dPurification=dPurification,
            maxiter=maxiter_heuristic,
            keep_classical_correlations=keep_classical_correlations,
        )
        return L, S, R, info
