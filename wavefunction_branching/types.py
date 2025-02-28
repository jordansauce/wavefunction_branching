# fmt: off
from typing import TypeAlias

from jaxtyping import Complex
from numpy.typing import NDArray

LeftSplittingTensor       : TypeAlias = Complex[NDArray, "nBranches dVirt_L dSlow"]
BlockDiagTensor           : TypeAlias = Complex[NDArray, "dPhys nBranches dSlow dSlow"]
RightSplittingTensor      : TypeAlias = Complex[NDArray, "nBranches dSlow dVirt_R"]
MatrixStack               : TypeAlias = Complex[NDArray, "dPhys dVirt_L dVirt_R"]
PurificationMatrixStack   : TypeAlias = Complex[NDArray, "nBranches dPhys dVirt_L dVirt_R"]

Matrix                   : TypeAlias = Complex[NDArray, "dVirt_L dVirt_R"]
SquareMatrix             : TypeAlias = Complex[NDArray, "dVirt dVirt"]
PurificationMPSTensor    : TypeAlias = Complex[NDArray, "dPhys dPurification dVirt dVirt"]
MPSTensor                : TypeAlias = Complex[NDArray, "dPhys dVirt dVirt"]
UnitarySplittingTensor   : TypeAlias = Complex[NDArray, "dVirt dSlow nBranches"] 
    # ^ where dSlow * nBranches = dVirt
SlowTensor               : TypeAlias = Complex[NDArray, "dPhys dSlow dSlow"] 
FastVector               : TypeAlias = Complex[NDArray, "dSlow"] 
LeftEnvironmentTensor    : TypeAlias = Complex[NDArray, "dPhys dVirt dPhys dVirt"] 
    # ^ From contracting an MPSTensor with its conjugate along the left virtual index
RightEnvironmentTensor   : TypeAlias = Complex[NDArray, "dPhys dVirt dPhys dVirt"] 
    # ^ From contracting an MPSTensor with its conjugate along the right_env virtual index


RhoLM                    : TypeAlias = Complex[NDArray, "nBranches dVirt_L nBranches dVirt_L"]
RhoMR                    : TypeAlias = Complex[NDArray, "nBranches dVirt_R nBranches dVirt_R"]
