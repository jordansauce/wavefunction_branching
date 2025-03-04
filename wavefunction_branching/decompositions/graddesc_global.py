# %%
# Note: this optimizes for total reconstructuion error, so that the branches sum to the full state, not just match expectation values on two-site regions

# TODO:
#    This should be generalized for situations where the modified region is larger than the desired size of non-interfering operators (eg. only requiring two-site non-inteference with an n-site branch region).
#   Currently the code implies n = m, where n is the size of the branch region, and m is the size of non-interfering operators.

# TODO:
#    Figure out why the norms don't add up even though we seem to be getting good reconstruction loss

# TODO:
#    Try the optimized decomposition of the reconstruction term (individual overlaps and norms)

# TODO:
#   Try stochastic gradient descent, where random vectors (or tensors) are contracted with the physical legs

# TODO:
#    Add some more validation metrics using eg. trace norms

from collections.abc import Iterable
from typing import Any, TypeAlias

import numpy as np
import torch
from jaxtyping import Complex, Float, Num
from opt_einops import (
    einsum,
    rearrange,
)  # use pip install git+https://github.com/jordansauce/opt_einops

# import pandas as pd
# import seaborn as sns
# from wavefunction_branching.types import MatrixStack
from wavefunction_branching.types import MatrixStack

TorchScalarFloat: TypeAlias = Float[torch.Tensor, ""]
TorchScalarComplex: TypeAlias = Complex[torch.Tensor, ""]
MatrixStackTorch: TypeAlias = Complex[torch.Tensor, "dPhys dVirt_L dVirt_R"]
MatrixStackAny: TypeAlias = Num[Any, "dPhys dVirt_L dVirt_R"]
ArrayLike: TypeAlias = np.ndarray | torch.Tensor


def to_torch(tensors: Iterable[np.ndarray]) -> list[torch.Tensor]:
    return [torch.tensor(t) for t in tensors]


def to_gpu(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    return [t.cuda() for t in tensors] if torch.cuda.is_available() else tensors


def to_cpu(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    return [t.cpu() for t in tensors]


def to_numpy(tensors: list[torch.Tensor]) -> list[np.ndarray]:
    return [t if isinstance(t, np.ndarray) else t.cpu().detach().numpy() for t in tensors]


def calc_norm(tensors: list[MatrixStackAny]):
    pattern = ", ".join([f"p{i} m{i} m{i + 1}" for i in range(len(tensors))])
    pattern += ", " + ", ".join(
        [
            f"p{i} m{'c' if i != 0 else ''}{i} m{'c' if i + 1 != len(tensors) else ''}{i + 1}"
            for i in range(len(tensors))
        ]
    )
    pattern += " -> "
    tensors_conj = [t.conj() for t in tensors]
    # print(f'calc_norm pattern: {pattern}')
    # p0 m0 m1, p1 m1 m2, p0 m0 mc1, p1 mc1 m2 ->
    return einsum(*tensors, *tensors_conj, pattern)  # type: ignore


def normalize(
    tensors: list[MatrixStackAny],
) -> list[MatrixStackAny]:
    norm = calc_norm(tensors)
    rescale_value = 1.0 / (norm ** (0.5 / len(tensors)))
    for t in tensors:
        t *= rescale_value
    return tensors


def contract_theta(tensors: list[MatrixStackAny]) -> TorchScalarComplex | np.ndarray:
    pattern = ", ".join([f"p{i} m{i} m{i + 1}" for i in range(len(tensors))])
    pattern += " -> " + " ".join([f"p{i}" for i in range(len(tensors))]) + f" m0 m{len(tensors)}"
    # print(f'contract_theta pattern: {pattern}')
    # p0 m0 m1, p1 m1 m2 -> p0 p1 m0 m2
    return einsum(*tensors, pattern).squeeze()  # type: ignore


def sum_squares(tensor: torch.Tensor) -> TorchScalarFloat:
    return (abs(tensor) ** 2).sum()


def calc_loss_reconstruction(
    tensors_orig: list[MatrixStackTorch],
    tensors_a: list[MatrixStackTorch],
    tensors_b: list[MatrixStackTorch],
) -> TorchScalarFloat:
    """this optimizes for total reconstructuion error, so that the branches sum to the full state, not just match expectation values on two-site regions"""
    theta_orig = contract_theta(tensors_orig)
    theta_a = contract_theta(tensors_a)
    theta_b = contract_theta(tensors_b)
    delta_theta = theta_a + theta_b - theta_orig
    return (abs(delta_theta) ** 2).sum()


def calc_loss_interference_middle(
    tensors_a: list[MatrixStackTorch], tensors_b: list[MatrixStackTorch]
) -> TorchScalarFloat:
    """The reduced density matrix on the middle region should not have interference cross-terms between the branches.
    Equivalently, there should be a joint record on the left and the right side of the middle region."""
    pattern = ", ".join([f"p{i} m{i} m{i + 1}" for i in range(len(tensors_a))])
    pattern += ", " + ", ".join(
        [
            f"pc{i} m{'c' if i != 0 else ''}{i} m{'c' if i + 1 != len(tensors_b) else ''}{i + 1}"
            for i in range(len(tensors_b))
        ]
    )
    pattern += " -> " + " ".join([f"p{i}" for i in range(len(tensors_a))])
    pattern += " " + " ".join([f"pc{i}" for i in range(len(tensors_b))])
    # print(f'calc_loss_interference_middle pattern: {pattern}')
    # p0 m0 m1, p1 m1 m2, pc0 m0 mc1, pc1 mc1 m2 -> p0 p1 pc0 pc1
    return sum_squares(einsum(*tensors_a, *[torch.conj(x) for x in tensors_b], pattern))  # type: ignore


def calc_loss_interference_right(
    tensors_a: list[MatrixStackTorch], tensors_b: list[MatrixStackTorch]
) -> TorchScalarFloat:
    """The reduced density matrix to the right, including all but one leftmostsite within the middle region,
    should not have any interference cross-terms between the branches.
    Equivalently, there should be a joint record on the left side of the middle region and the
    leftmost tensor within the middle region"""
    pattern = ", ".join([f"p{i} m{i} m{i + 1}" for i in range(len(tensors_a))])
    pattern += ", " + ", ".join(
        [
            f"p{'c' if i != 0 else ''}{i} m{'c' if i != 0 else ''}{i} mc{i + 1}"
            for i in range(len(tensors_b))
        ]
    )
    pattern += " -> " + " ".join([f"p{i}" for i in range(len(tensors_a)) if i > 0])
    pattern += " " + " ".join([f"pc{i}" for i in range(len(tensors_b)) if i > 0])
    pattern += f" m{len(tensors_a)} mc{len(tensors_b)}"
    # print(f'calc_loss_interference_right pattern: {pattern}')
    # p0 m0 m1, p1 m1 m2, p0 m0 mc1, pc1 mc1 mc2 -> p1 pc1 m2 mc2
    return sum_squares(einsum(*tensors_a, *[torch.conj(x) for x in tensors_b], pattern))  # type: ignore


def calc_loss_interference_left(
    tensors_a: list[MatrixStackTorch], tensors_b: list[MatrixStackTorch]
) -> TorchScalarFloat:
    """The reduced density matrix to the left, including all but one rightmost site within the middle region,
    should not have any interference cross-terms between the branches.
    Equivalently, there should be a joint record on the right side of the middle region and the
    rightmost tensor within the middle region"""
    pattern = ", ".join([f"p{i} m{i} m{i + 1}" for i in range(len(tensors_a))])
    pattern += ", " + ", ".join(
        [
            f"p{'c' if i + 1 != len(tensors_b) else ''}{i} mc{i} m{'c' if i + 1 != len(tensors_b) else ''}{i + 1}"
            for i in range(len(tensors_b))
        ]
    )
    pattern += " -> " + " ".join([f"p{i}" for i in range(len(tensors_a) - 1)])
    pattern += " " + " ".join([f"pc{i}" for i in range(len(tensors_b) - 1)])
    pattern += " m0 mc0"
    # print(f'calc_loss_interference_left pattern: {pattern}')
    # p0 m0 m1, p1 m1 m2, pc0 mc0 mc1, p1 mc1 m2 -> p0 pc0 m0 mc0
    return sum_squares(einsum(*tensors_a, *[torch.conj(x) for x in tensors_b], pattern))  # type: ignore


def calc_loss_interference(
    tensors_a: list[MatrixStackTorch], tensors_b: list[MatrixStackTorch]
) -> TorchScalarFloat:
    loss_interference_middle = calc_loss_interference_middle(tensors_a, tensors_b)
    loss_interference_left = calc_loss_interference_left(tensors_a, tensors_b)
    loss_interference_right = calc_loss_interference_right(tensors_a, tensors_b)
    return loss_interference_middle + loss_interference_right + loss_interference_left


def calc_loss(
    tensors_orig: list[MatrixStackTorch],
    tensors_a: list[MatrixStackTorch],
    tensors_b: list[MatrixStackTorch],
    reconstruction_weight: float = 1.0,
    interference_weight: float = 1.0,
) -> TorchScalarFloat:
    return reconstruction_weight * calc_loss_reconstruction(
        tensors_orig, tensors_a, tensors_b
    ) + interference_weight * calc_loss_interference(tensors_a, tensors_b)


def optimize(
    tensors_orig_np: Iterable[MatrixStack],
    tensors_a_np: Iterable[MatrixStack],
    tensors_b_np: Iterable[MatrixStack],
    lr: float = 0.005,
    reconstruction_weight: float = 1.0,
    interference_weight: float = 1.0,
    gamma: float = 0.95,
    epochs: int = 20,
    steps_per_epoch: int = 50,
) -> tuple[list[MatrixStack], list[MatrixStack], dict[str, list[float]]]:
    losses = []
    reconstruction_losses = []
    interference_losses = []

    tensors_orig = to_gpu(to_torch(tensors_orig_np))
    tensors_a = to_gpu(to_torch(tensors_a_np))
    tensors_b = to_gpu(to_torch(tensors_b_np))

    # Calculate initial loss
    with torch.no_grad():
        reconstruction_loss = calc_loss_reconstruction(tensors_orig, tensors_a, tensors_b).item()
        interference_loss = calc_loss_interference(tensors_a, tensors_b).item()
        reconstruction_losses.append(reconstruction_loss)
        interference_losses.append(interference_loss)
        losses.append(
            reconstruction_weight * reconstruction_loss + interference_weight * interference_loss
        )

    parameters = [*tensors_a, *tensors_b]
    parameters = [t.requires_grad_(True) for t in parameters]

    optimizer = torch.optim.Adam(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    for epoch in range(epochs):
        assert steps_per_epoch > 0
        for i in range(steps_per_epoch):
            optimizer.zero_grad()
            reconstruction_loss = calc_loss_reconstruction(tensors_orig, tensors_a, tensors_b)
            interference_loss = calc_loss_interference(tensors_a, tensors_b)
            loss = (
                reconstruction_weight * reconstruction_loss
                + interference_weight * interference_loss
            )
            # norm_a = calc_norm(tensors_a)
            # norm_b = calc_norm(tensors_b)
            # norm_loss = (norm_a - 0.5)**2 + (norm_b - 0.5)**2
            # loss += norm_loss
            loss.backward()
            optimizer.step()
        assert isinstance(loss, torch.Tensor)
        losses.append(loss.item())
        assert isinstance(reconstruction_loss, torch.Tensor)
        assert isinstance(interference_loss, torch.Tensor)
        reconstruction_losses.append(reconstruction_loss.item())
        interference_losses.append(interference_loss.item())
        # print(f'{epoch+1}/{epochs} | {i+1}/{steps_per_epoch} | loss: {losses[-1]} | reconstruction loss = {reconstruction_losses[-1]} | interference loss = {interference_losses[-1]}')
        scheduler.step()
    tensors_a_np = to_numpy(tensors_a)
    tensors_b_np = to_numpy(tensors_b)
    return (
        tensors_a_np,
        tensors_b_np,
        dict(
            losses=losses,
            reconstruction_losses=reconstruction_losses,
            interference_losses=interference_losses,
        ),
    )


def split(
    tensor: Num[Any, "pl pr l r"], form_L=0.5, form_R=0.5, cutoff=0.0, max_bond=None
) -> tuple[Num[Any, "p l m"], Num[Any, "p m r"]]:
    N = tensor.shape[0]
    p = int(np.sqrt(N))
    M = rearrange(tensor, "(pl pr) l r -> (pl l) (pr r)", pl=p)
    L, svals, R = np.linalg.svd(M, full_matrices=False)
    n_svals = int(round(np.sum(svals > cutoff)))
    if max_bond is not None:
        n_svals = int(min(n_svals, max_bond))
    if cutoff > 0.0 or max_bond is not None:
        svals = svals[:n_svals]
        L = L[:, :n_svals]
        R = R[:n_svals, :]

    L = L @ np.diag(svals**form_L)
    R = np.diag(svals**form_R) @ R

    L = rearrange(L, "(p l) m -> p l m", p=p)
    R = rearrange(R, "m (p r) -> p m r", p=p)
    return L, R


# %%
