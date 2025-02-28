import numpy as np

from wavefunction_branching.types import Matrix, MPSTensor


def make_square(tensor: MPSTensor, xFast: int = 2) -> MPSTensor:
    shape = list(tensor.shape)
    tensor = tensor.reshape(-1, shape[-2], shape[-1])  # physical, left, right
    # Ensure that the tensors are square
    if shape[-2] != shape[-1] or max(tensor.shape[1:]) % xFast != 0:
        shape = list(tensor.shape)
        chi = max(shape[1:])
        if chi % xFast != 0:  # Make the bond dimension divisible by the number of blocks
            chi += xFast - chi % xFast
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
