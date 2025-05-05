"""Functions for measuring and plotting properties of wavefunctions and density matrices."""

import copy
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quimb.tensor as qtn
import scipy as sp
import seaborn as sns
from jaxtyping import Complex
from opt_einops import einsum as opt_einops_einsum
from opt_einops import rearrange


def einsum(*args, **kwargs):
    return opt_einops_einsum(*args, **kwargs)


CURRENT_PATH = Path(__file__).parent.absolute()
PLOTS_DIR = CURRENT_PATH / f"plots/{datetime.now().strftime('%Y-%m-%d')}"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)


def calculate_TM_eigs(
    T_: Complex[np.ndarray, "p l r"],
    left=True,
    verbose=False,
    n=None,
    maxiter=None,
    tol=1e-5,
    return_eigenvectors=False,
    which="LM",
):
    """Calculate the eigenvalues and eigenmatrices of the Transfer matrix formed out of AB and AB*
    INPUTS: T - a tensor with legs (physical, bond_left, bond_right)"""

    T = copy.deepcopy(T_)
    # Expand dimensions if T is not square
    matrix_shapes = T.shape[1:]
    if matrix_shapes[0] != matrix_shapes[1]:
        max_dim = max(matrix_shapes)
        expanded = np.zeros((T.shape[0], matrix_shapes[0], matrix_shapes[1]), dtype=T.dtype)
        expanded[:, : matrix_shapes[0], : matrix_shapes[1]] = T
        T = expanded

    AB = qtn.Tensor(T, inds=["p", "al", "br"])  # type: ignore

    # Get eigenmatrices of the transfer matrix
    AB_H = AB.H
    assert AB_H is not None
    TM = AB & AB_H.reindex({"al": "al*", "br": "br*"})
    if n is None:
        n = min(T.shape[0:]) ** 2

    class Tmat_LinearOperator(sp.sparse.linalg.LinearOperator):
        def __init__(self, TM, left=True):
            assert set(TM.outer_inds()) == set(["al", "al*", "br", "br*"])
            self.TM = TM
            self.left = left
            self.dtype = np.dtype(complex)
            self.shape = (
                self.TM.ind_size("al") * self.TM.ind_size("al*"),
                self.TM.ind_size("br") * self.TM.ind_size("br*"),
            )
            if not left:
                self.shape = self.shape[::-1]

        def _transpose(self):
            return self.TM.reindex({"br": "al", "br*": "al*", "al": "br", "al*": "br*"})

        def _conj(self):
            return self.TM.conj()

        def _adjoint(self):
            TM_temp = self._transpose()
            return TM_temp.conj().reindex({"br": "al", "br*": "al*", "al": "br", "al*": "br*"})

        def _matvec(self, vec):
            # vec is a 1d np array of shape (TM.ind_size('al')*TM.ind_size('al*'))
            if self.left:
                TM = self.TM
            else:
                TM = self.transpose()
            tvec = qtn.Tensor(inds=["al2"], data=vec)
            tvec.unfuse(
                {"al2": ["al", "al*"]},
                {"al2": [self.TM.ind_size("al"), self.TM.ind_size("al*")]},
                inplace=True,
            )
            TM_tvec = (TM & tvec).contract().reindex({"br": "al", "br*": "al*"})
            TM_tvec.fuse({"al2": ["al", "al*"]})
            return TM_tvec.data

    TM_linop = Tmat_LinearOperator(TM, left=left)
    eigs = sp.sparse.linalg.eigs(
        TM_linop,
        k=n,
        which=which,
        maxiter=maxiter,
        tol=tol,
        return_eigenvectors=return_eigenvectors,
    )
    if not return_eigenvectors:
        return np.sort(eigs)[::-1], None
    else:
        TM_evals, TM_evecs = eigs
        sorted_inds = np.argsort([np.real(e) for e in TM_evals])[::-1]
        trunc_sorted_inds = sorted_inds
        if len(sorted_inds) > n:
            trunc_sorted_inds = sorted_inds[:n]
        TM_evecs = TM_evecs[:, trunc_sorted_inds]
        TM_evals = TM_evals[trunc_sorted_inds]
        TM_emats = rearrange(TM_evecs, "( l r ) e -> e l r", l=AB.ind_size("al"))
        for i in range(TM_emats.shape[0]):
            rephase = np.average(
                TM_emats[i] / np.conj(np.transpose(TM_emats[i])), weights=np.abs(TM_emats[i])
            )
            if abs(np.sqrt(rephase)) > 0:
                TM_emats[i] /= np.sqrt(rephase)

        if verbose:
            for i in range(min(2, len(TM_evals))):
                print(f"TM eig {i}: value = {TM_evals[i]}, mat = ")
                plt.imshow(abs(TM_emats[i]))
                plt.show()
                # Check Hermitian:
                print(
                    f"TM_emats[{i}] is hermitian: {np.allclose(np.conj(np.transpose(TM_emats[i])), TM_emats[i])}"
                )
        return TM_evals, TM_emats


def get_rho_2site_orig(
    L: Complex[np.ndarray, "lc l"],
    AB: Complex[np.ndarray, "l pa pb r"],
    R: Complex[np.ndarray, "r rc"],
) -> tuple[Complex[np.ndarray, "pa pb pac pbc"], float]:
    rho = einsum(L, AB, AB.conj(), R, "lc l, l pa pb r , lc pac pbc rc, r rc -> pa pb pac pbc")
    rho_norm = einsum(rho, "pa pb pa pb ->")
    return rho / rho_norm, rho_norm


##############################^^^ get_rho_2site_orig() ^^^##########################################


def get_rho_2site_purified(
    L: Complex[np.ndarray, "Lc L"],
    U_purified: Complex[np.ndarray, "i L l"],
    AB_purified: Complex[np.ndarray, "i p l r"],
    Vh_purified: Complex[np.ndarray, "i r R"],
    R: Complex[np.ndarray, "R Rc"],
    diag_only=False,
) -> tuple[Complex[np.ndarray, "pa pb pac pbc"], complex]:
    if diag_only == True:
        j = "i"
    else:
        j = "j"
    rho = einsum(
        L,
        U_purified,
        AB_purified,
        Vh_purified,
        U_purified.conj(),
        AB_purified.conj(),
        Vh_purified.conj(),
        R,
        "Lc L, i  L  l,   i  p  l  r ,  i r  R ,"
        + f"{j} Lc lc, {j} pc lc rc, {j} rc Rc, R Rc -> p pc ",
    )
    rho = rearrange(rho, "(pa pb) (pac pbc) -> pa pb pac pbc", pa=2, pac=2)
    rho_norm = einsum(rho, "pa pb pa pb ->")
    return rho / rho_norm, rho_norm


##############################^^^ get_rho_2site_purified() ^^^######################################


def get_rho_4site_orig(
    L: Complex[np.ndarray, "lc l"],
    AB: Complex[np.ndarray, "l pa pb r"],
    λ_inv: Complex[np.ndarray, "r"],
    R: Complex[np.ndarray, "r rc"],
) -> tuple[Complex[np.ndarray, "pa1 pb1 pa2 pb2 pac1 pbc1 pac2 pbc2"], complex]:
    rho = einsum(
        L,
        AB,
        λ_inv,
        AB,
        AB.conj(),
        λ_inv.conj(),
        AB.conj(),
        R,
        #  L,        AB,      λ_inv,     AB
        "lc l, l  pa1  pb1  m , m,  m  pa2  pb2  r , "
        #        AB.conj(), λ_inv.conj(), AB.conj(), R
        + "lc pac1 pbc1 mc, mc, mc pac2 pbc2 rc, r rc"
        + " -> pa1 pb1 pa2 pb2 pac1 pbc1 pac2 pbc2",
    )
    rho_norm = einsum(rho, "pa1 pb1 pa2 pb2 pa1 pb1 pa2 pb2 ->")
    return rho / rho_norm, rho_norm


##############################^^^ get_rho_4site_orig() ^^^##########################################


def get_rho_4site_purified(
    L: Complex[np.ndarray, "Lc L"],
    U_purified: Complex[np.ndarray, "i L l"],
    AB_purified: Complex[np.ndarray, "i p l r"],
    Vh_purified: Complex[np.ndarray, "i r R"],
    λ_inv: Complex[np.ndarray, "R"],
    R: Complex[np.ndarray, "R Rc"],
    diag_only=False,
) -> tuple[Complex[np.ndarray, "pa1 pb1 pa2 pb2 pac1 pbc1 pac2 pbc2"], complex]:
    if diag_only == True:
        j1 = "i1"
        j2 = "i2"
    else:
        j1 = "j1"
        j2 = "j2"
    rho_grouped = einsum(
        L,
        U_purified,
        AB_purified,
        Vh_purified,
        λ_inv,
        U_purified,
        AB_purified,
        Vh_purified,
        U_purified.conj(),
        AB_purified.conj(),
        Vh_purified.conj(),
        λ_inv.conj(),
        U_purified.conj(),
        AB_purified.conj(),
        Vh_purified.conj(),
        R,
        # L   |     U    |       AB      |    Vh    | λ |     U    |     AB        |  Vh     |
        "Lc L , i1 L  l1 , i1 p1  l1  r1 , i1 r1  M , M , i2 M  l2 , i2 p2  l2  r2 , i2 r2  R,  "
        #           Uc   |         ABc     |     Vh     | λc |      Uc   |          ABc    |     Vhc    | R
        + f" {j1} Lc lc1, {j1} pc1 lc1 rc1, {j1} rc1 Mc, Mc, {j2} Mc lc2, {j2} pc2 lc2 rc2, {j2} rc2 Rc, R Rc "
        + " -> p1 p2 pc1 pc2",
    )
    p = 2
    rho_ungrouped = rearrange(
        rho_grouped,
        "(pa1 pb1) (pa2 pb2) (pac1 pbc1) (pac2 pbc2) -> pa1 pb1 pa2 pb2 pac1 pbc1 pac2 pbc2",
        pa1=p,
        pa2=p,
        pac1=p,
        pac2=p,
    )
    rho_norm = einsum(rho_ungrouped, "pa1 pb1 pa2 pb2 pa1 pb1 pa2 pb2 ->")
    return rho_ungrouped / rho_norm, rho_norm


##############################^^^ get_rho_4site_purified() ^^^######################################


def trace_norm(
    rho,  # [Complex[np.ndarray, "pa pb ... pac pbc ..."],
) -> float:
    """Calculate the trace norm of a density matrix"""
    n_phys_inds = len(rho.shape) // 2
    shape = (np.prod(rho.shape[:n_phys_inds]), np.prod(rho.shape[n_phys_inds:]))
    rho_flat = np.reshape(rho, shape)
    svals = np.linalg.svd(rho_flat, full_matrices=False, compute_uv=False)
    trace_norm = abs(np.sum(svals))
    assert isinstance(trace_norm, float)
    return trace_norm


def trace(rho):
    n_phys_inds = len(rho.shape) // 2
    shape = (np.prod(rho.shape[:n_phys_inds]), np.prod(rho.shape[n_phys_inds:]))
    rho_flat = np.reshape(rho, shape)
    return np.trace(rho_flat)


def trace_distance(
    rho_1,  # [Complex[np.ndarray, "pa pb ... pac pbc ..."],
    rho_2,  # [Complex[np.ndarray, "pa pb ... pac pbc ..."]
    normalize=False,
) -> float:
    """Calculate the trace norm of the difference between two density matrices, which is a measure of fidelity,
    bounding the difference in expectation values between all observables on those density matrices
    trace_norm = sum_svals(rho_2 - rho_1)
    """
    if normalize:
        rho_1 = rho_1 / trace(rho_1)
        rho_2 = rho_2 / trace(rho_2)
    delta_rho = rho_2 - rho_1
    return trace_norm(delta_rho)


#########################^^^ trace_norm()  trace()  trace_distance() ^^^############################


def get_rho_half_LM(matrix_stack: Complex[np.ndarray, "p l r"]) -> Complex[np.ndarray, "p l pc lc"]:
    p, L, R = matrix_stack.shape
    matrix_stack_isometric_L, svals, Vh = np.linalg.svd(
        rearrange(matrix_stack, "p L R -> (p L) R"), full_matrices=False
    )
    matrix_stack_isometric_L = rearrange(matrix_stack_isometric_L, "(p L) R -> p L R", p=p, L=L)
    rho_half_LM = einsum(
        matrix_stack_isometric_L,
        svals,
        np.conj(matrix_stack_isometric_L),
        "p L R, R, pc Lc R -> p L pc Lc",
    )
    return rho_half_LM


def get_rho_half_MR(matrix_stack: Complex[np.ndarray, "p l r"]) -> Complex[np.ndarray, "p r pc rc"]:
    transpose = rearrange(matrix_stack, "p L R -> p R L")
    return get_rho_half_LM(transpose)


def LMR_trace_distances(
    orig: Complex[np.ndarray, "p l r"],
    purif: Complex[np.ndarray, "b p l r"],
    measure_LR=False,
    measure_split=False,
):
    """Measure various trace distances between a ``theta'' wavefunction and a purification mixed-state theta wavefunction"""
    purif = purif[:, : orig.shape[0], : orig.shape[1], : orig.shape[2]]

    norm_orig = np.sqrt(abs(orig.flatten() @ np.conj(orig.flatten())))
    norm_purif_mixed = np.sum(
        np.sqrt(np.abs(einsum(purif, np.conj(purif), "b p l r, b p l r -> b")))
    )  # np.sqrt(abs(purif.flatten() @ np.conj(purif.flatten())))
    purif_pure = np.sum(copy.deepcopy(purif), 0)
    norm_purif_pure = np.sqrt(abs(einsum(purif_pure, np.conj(purif_pure), "p l r, p l r -> ")))
    purif_pure = purif_pure / norm_purif_pure

    orig = orig / norm_orig

    overlap_pure = abs(einsum(purif_pure, np.conj(orig), "p l r, p l r -> "))

    purif = purif / norm_purif_mixed
    overlap_mixed = np.sqrt(
        abs(
            einsum(
                purif,
                np.conj(orig),
                np.conj(purif),
                orig,
                "b p l r, p l r, b p2 l2 r2, p2 l2 r2 -> ",
            )
        )
    )

    out = {
        "norm_orig": norm_orig,
        "norm_purif_mixed": norm_purif_mixed,
        "norm_purif_pure": norm_purif_pure,
        "norm_error_pure": abs(norm_orig - norm_purif_pure) / norm_orig,
        "norm_error_mixed": abs(norm_orig - norm_purif_mixed) / norm_orig,
        "one_minus_overlap_pure": abs(1.0 - overlap_pure),
        "one_minus_overlap_mixed": abs(1.0 - overlap_mixed),
        "global_reconstruction_error_trace_distance": np.sqrt(np.abs(1.0 - overlap_pure**2)),
    }

    # rho_half_LM_orig = get_rho_half_LM(orig)
    # rho_half_LM_purif = get_rho_half_LM(rearrange(purif, 'b p l r -> p l (b r)'))

    rho_LM_orig = einsum(orig, np.conj(orig), "p l r,   pc lc r -> p l pc lc")
    rho_LM_purif = einsum(purif, np.conj(purif), "b p l r, b pc lc r -> p l pc lc")
    out["trace_distance_LM"] = trace_distance(rho_LM_orig, rho_LM_purif, normalize=True)

    rho_MR_orig = einsum(orig, np.conj(orig), "p l r,   pc l rc -> p r pc rc")
    rho_MR_purif = einsum(purif, np.conj(purif), "b p l r, b pc l rc -> p r pc rc")
    out["trace_distance_MR"] = trace_distance(rho_MR_orig, rho_MR_purif, normalize=True)

    if measure_LR:
        # Disabled by default as this is a chi^4 operation
        rho_LR_orig = einsum(orig, np.conj(orig), "p l r,   p lc rc -> l r lc rc")
        rho_LR_purif = einsum(purif, np.conj(purif), "b p l r, b p lc rc -> l r lc rc")
        out["trace_distance_LR"] = trace_distance(rho_LR_orig, rho_LR_purif, normalize=True)

    rho_L_orig = einsum(rho_LM_orig, "p l p lc -> l lc")
    rho_L_purif = einsum(rho_LM_purif, "p l p lc -> l lc")
    out["trace_distance_L"] = trace_distance(rho_L_orig, rho_L_purif, normalize=True)

    rho_R_orig = einsum(rho_MR_orig, "p r p rc -> r rc")
    rho_R_purif = einsum(rho_MR_purif, "p r p rc -> r rc")
    out["trace_distance_R"] = trace_distance(rho_R_orig, rho_R_purif, normalize=True)

    rho_M_orig = einsum(rho_LM_orig, "p l pc l -> p pc")
    rho_M_purif = einsum(rho_LM_purif, "p l pc l -> p pc")
    out["trace_distance_M"] = trace_distance(rho_M_orig, rho_M_purif, normalize=True)

    if measure_split:
        orig_expanded = rearrange(
            orig, "  (p1 p2) l r ->   p1 p2 l r", p1=int(np.sqrt(orig.shape[0]))
        )
        purif_expanded = rearrange(
            purif, "b (p1 p2) l r -> b p1 p2 l r", p1=int(np.sqrt(orig.shape[0]))
        )

        rho_Lpl_orig = einsum(
            orig_expanded, np.conj(orig_expanded), "pl pr l r,   plc pr lc r -> pl l plc lc"
        )
        rho_Lpl_purif = einsum(
            purif_expanded, np.conj(purif_expanded), "b pl pr l r, b plc pr lc r -> pl l plc lc"
        )
        out["trace_distance_Lpl"] = trace_distance(rho_Lpl_orig, rho_Lpl_purif, normalize=True)

        rho_Rpr_orig = einsum(
            orig_expanded, np.conj(orig_expanded), "pl pr l r,   pl prc l rc -> pr r prc rc"
        )
        rho_Rpr_purif = einsum(
            purif_expanded, np.conj(purif_expanded), "b pl pr l r, b pl prc l rc -> pr r prc rc"
        )
        out["trace_distance_Rpr"] = trace_distance(rho_Rpr_orig, rho_Rpr_purif, normalize=True)

    return out


##############################^^^ measure_trace_distances() ^^^##################################################


def calculate_properties_2site_density_matrix(rho_: Complex[np.ndarray, "pa pb pac pbc"]):
    """
    Inputs:
        rho: a two-site physical density matrix of shape (p_a  p_b  p_a*  p_b*),
            where p are physical indices.
            rho may be grabbed from a tenpy wavefunction psi like this:
            AB = psi.get_theta(0,n=2, formL=1, formR=1).to_ndarray()
            rho = einsum(
                L, AB, AB.conj(), R,
                'lc l, l pa pb r , lc pac pbc rc, r rc -> pa pb pac pbc'
            )
    """
    props = {}
    norm = einsum(rho_, "pa pb pa pb -> ")
    if norm > 0.0:
        rho = rho_ / norm
    else:
        rho = rho_

    props["tr(rho)"] = norm
    props["rho_trace_norm"] = trace_norm(rho)

    # single-site operators
    σx = np.array([[0, 1], [1, 0]])
    σy = np.array([[0, -1j], [1j, 0]])
    σz = np.array([[1, 0], [0, -1]])
    single_site_operators = {"σx": σx, "σy": σy, "σz": σz}

    # Single-site expectation values
    for op_name, op in single_site_operators.items():
        expectation_a = float(einsum(rho, op, "pa pb pac pb, pa pac -> ").real)
        expectation_b = float(einsum(rho, op, "pa pb pa pbc, pb pbc -> ").real)
        props["〈" + op_name + "A" + "〉"] = expectation_a
        props["〈" + op_name + "B" + "〉"] = expectation_b
        props["〈" + op_name + "〉"] = (expectation_a + expectation_b) / 2.0

    # Two-site expectation values
    for op1_name, op1 in single_site_operators.items():
        for op2_name, op2 in single_site_operators.items():
            expectation_ab = float(einsum(rho, op1, op2, "pa pb pac pbc, pa pac, pb pbc -> ").real)
            if op1_name != op2_name:
                expectation_ba = float(
                    einsum(rho, op2, op1, "pa pb pac pbc, pa pac, pb pbc -> ").real
                )
                props["〈" + op1_name + "A " + op2_name + "B〉"] = expectation_ab
                props["〈" + op2_name + "A " + op1_name + "B" + "〉"] = expectation_ba
                props["〈" + op1_name + " " + op2_name + "〉"] = (
                    expectation_ab + expectation_ba
                ) / 2.0
            else:
                props["〈" + op1_name + " " + op2_name + "〉"] = expectation_ab
    return props


##############################^^^ calculate_properties_2site_density_matrix() ^^^###################


def calculate_properties_4site_density_matrix(
    rho_: Complex[np.ndarray, "pa1 pb1 pa2 pb2 pac1 pbc1 pac2 pbc2"],
):
    """
    Inputs:
        rho: a two-site physical density matrix of shape (p_a1  p_b1  p_a2  p_b2  p_a1*  p_b1*  p_a2*  p_b2* )
    """
    props = {}
    norm = einsum(rho_, "pa1 pb1 pa2 pb2 pa1 pb1 pa2 pb2 ->")
    rho = rho_ / norm

    # single-site operators
    σx = np.array([[0, 1], [1, 0]])
    σy = np.array([[0, -1j], [1j, 0]])
    σz = np.array([[1, 0], [0, -1]])
    single_site_operators = {"σx": σx, "σy": σy, "σz": σz}

    # Distance 3 correlators
    for op1_name, op1 in single_site_operators.items():
        for op2_name, op2 in single_site_operators.items():
            expectation_a = float(
                einsum(
                    rho, op1, op2, "pa1 pb1 pa2 pb2 pac1 pb1 pac2 pb2, pa1 pac1, pa2 pac2 -> "
                ).real
            )
            expectation_b = float(
                einsum(
                    rho, op1, op2, "pa1 pb1 pa2 pb2 pa1 pbc1 pa2 pbc2, pb1 pbc1, pb2 pbc2 -> "
                ).real
            )
            props["〈" + op1_name + "A" + " 1 " + op2_name + "A〉"] = expectation_a
            props["〈" + op1_name + "B" + " 1 " + op2_name + "B〉"] = expectation_b
            props["〈" + op1_name + " 1 " + op2_name + "〉"] = (expectation_a + expectation_b) / 2.0

    # Distance 4 correlators
    for op1_name, op1 in single_site_operators.items():
        for op2_name, op2 in single_site_operators.items():
            expectation_ab = float(
                einsum(
                    rho, op1, op2, "pa1 pb1 pa2 pb2 pac1 pb1 pa2 pbc2, pa1 pac1, pb2 pbc2 -> "
                ).real
            )
            props["〈" + op1_name + "A" + " 2 " + op2_name + "B〉"] = expectation_ab
    return props


##############################^^^ calculate_properties_4site_density_matrix() ^^^###################


def plot_properties_rho_comparison(props_dicts: list[dict], labels: list[str], filename=None):
    expectation_type = []
    expectation_value = []
    label = []
    for i in range(len(props_dicts)):
        for key in props_dicts[i]:
            expectation_type.append(
                key.replace("〈", "").replace("〉", "").replace(" ", "").replace("σ", "")
            )
            expectation_value.append(float(np.real(props_dicts[i][key])))
            label.append(labels[i])
    props = {
        "expectation_type": expectation_type,
        "expectation_value": expectation_value,
        "label": label,
    }

    df = pd.DataFrame(props)
    sns.set_theme(style="whitegrid", rc={"figure.figsize": (20, 6)})
    palette = sns.color_palette("Paired")
    palette = [
        palette[1],
        palette[4],
        palette[6],
        palette[2],
        palette[8],
        palette[2],
        palette[3],
        palette[5],
        palette[7],
        palette[3],
        palette[9],
    ]
    b = sns.barplot(df, x="expectation_type", y="expectation_value", hue="label", palette=palette)
    b.set_xticklabels(b.get_xticklabels(), size=7)
    if filename:
        plt.savefig(PLOTS_DIR / filename)
    plt.show()


##############################^^^ plot_props() ^^^##################################################


def measure_psi(psi, coarsegrain_from=None, coarsegrain_size=2, return_TM_eigs=False, check=True):
    """Compute various properties of psi, and store them in the returned dict"""
    if coarsegrain_from is None:
        coarsegrain_from = (psi.L - coarsegrain_size) // 2

    # psi.test_sanity()
    # psi.canonical_form()

    props = {}
    bond_dimensions = psi.chi
    bc_MPS = psi.bc
    entanglement_entropies = psi.entanglement_entropy()
    props["bc"] = bc_MPS
    props["norm"] = psi.norm
    props["prob"] = np.abs(psi.norm) ** 2
    props["bond dimensions"] = bond_dimensions
    props["entanglement entropies"] = entanglement_entropies
    props["average bond dimension"] = np.mean(bond_dimensions)
    props["average entanglement entropy"] = np.mean(entanglement_entropies)
    props["max bond dimension"] = max(bond_dimensions)
    props["max entanglement entropy"] = max(entanglement_entropies)

    # for bond in range(psi.L-1):
    coarsegrain_from_2site = (psi.L - 2) // 2
    if (coarsegrain_from_2site > 0 and coarsegrain_from_2site + 2 < psi.L) or psi.bc != "finite":
        sites_2site = np.arange(coarsegrain_from_2site, coarsegrain_from_2site + 2)
        rho_2site = psi.get_rho_segment(sites_2site).to_ndarray()
        # rho_2site = einsum(theta_2site, np.conj(theta_2site), 'L p1 p2 R, L pc1 pc2 R -> p1 p2 pc1 pc2')
        # rho_2site = psi.get_rho_segment(range(coarsegrain_from, coarsegrain_from+2)).to_ndarray()
        props.update(calculate_properties_2site_density_matrix(rho_2site))
    coarsegrain_from_4site = (psi.L - 4) // 2
    if (coarsegrain_from_4site > 0 and coarsegrain_from_4site + 4 < psi.L) or psi.bc != "finite":
        sites_4site = np.arange(coarsegrain_from_4site, coarsegrain_from_4site + 4)
        rho_4site = psi.get_rho_segment(sites_4site).to_ndarray()
        # theta_4site = psi.get_theta(coarsegrain_from_4site, n=4, cutoff=0.0, formL=1.0, formR=1.0).to_ndarray()
        # rho_4site = einsum(theta_4site, np.conj(theta_4site), 'L p1 p2 p3 p4 R, L pc1 pc2 pc3 pc4 R -> p1 p2 p3 p4 pc1 pc2 pc3 pc4')
        # rho_4site = psi.get_rho_segment(range(coarsegrain_from, coarsegrain_from+4)).to_ndarray()
        props.update(calculate_properties_4site_density_matrix(rho_4site))

    coarsegrain_from_6site = (psi.L - 6) // 2
    if (coarsegrain_from_6site > 0 and coarsegrain_from_6site + 6 < psi.L) or psi.bc != "finite":
        # sites_6site = list(np.arange(coarsegrain_from_6site, coarsegrain_from_6site+6))
        # sites_6site = sites_6site[:2] + sites_6site[-2:]
        # rho_6site = psi.get_rho_segment(sites_6site).to_ndarray()
        theta_6site = psi.get_theta(
            coarsegrain_from_6site, n=6, cutoff=0.0, formL=1.0, formR=1.0
        ).to_ndarray()
        rho_6site = einsum(
            theta_6site,
            np.conj(theta_6site),
            "L p1 p2 p3 p4 p5 p6 R, L pc1 pc2 p3 p4 pc5 pc6 R -> p1 p2 p5 p6 pc1 pc2 pc5 pc6",
        )
        props_6site = calculate_properties_4site_density_matrix(rho_6site)
        props_6site_renamed = {}
        for key, value in props_6site.items():
            props_6site_renamed[key.replace("1", "3").replace("2", "4")] = value
        props.update(props_6site_renamed)

    props["Id"] = np.mean(psi.expectation_value("Id"))

    # print(f'Measuring psi - norm = {psi.norm} - max entanglement entropy = {props["max entanglement entropy"]} - chi = {psi.chi} ')

    if return_TM_eigs:
        print("Calculating TM eigenspectrum")
        AB = psi.get_theta(coarsegrain_from, n=coarsegrain_size, formL=1.0, formR=0.0).to_ndarray()
        AB = rearrange(AB, "L ... R -> (...) L R")
        # Transfer matrix spectrum and correlation length
        if not all([chi > 2 for chi in psi.chi]):
            props["correlation length"] = 0.0
            # props['dominant TM singular value'] = 1.0
            # props['second-dominant TM singular value magnitude'] = 0.0
            # props['correlation length from svals'] = 0.0
            props["dominant TM eigenvalue"] = 1.0
            props["second-dominant TM eigenvalue magnitude"] = 0.0
            props["second-dominant TM eigenvalue phase"] = 0.0
            props["second-dominant TM eigenvalue real part"] = 0.0
            props["second-dominant TM eigenvalue imag part"] = 0.0
        else:
            # # Transfer matrix singular values
            # ABAB = einsum(AB, np.conj(AB), 'pl L r, pr l r  -> pl pr L l')
            # ABAB = rearrange(ABAB, 'pl pr L l -> (pl pr) L l')
            # print(f'ABAB.shape = {ABAB.shape}')
            # t0_mine = time.time()
            # TM_svals, TM_smats = calculate_TM_eigs(ABAB, n=2)
            # print(f'    TM_svals raw = {TM_svals}')
            # if abs(TM_svals[1]) > abs(TM_svals[0]):
            #     TM_svals = TM_svals[::-1]
            # TM_svals /= TM_svals[0]
            # TM_svals = np.sqrt(np.abs(np.array(TM_svals)))
            # tf_mine = time.time()
            # print(f'    time taken = {tf_mine-t0_mine}')
            # print(f'    TM_svals = {TM_svals}')
            # props['dominant TM singular value'] = TM_svals[0]
            # props['second-dominant TM singular value'] = TM_svals[1]
            # props['correlation length from svals'] = 2/np.log(TM_svals[0]/TM_svals[1])

            # Transfer matrix eigenvalues
            print(f"AB.shape = {AB.shape}")
            t0_mine = time.time()
            TM_evals, TM_emats = calculate_TM_eigs(AB, n=2)
            if abs(TM_evals[1]) > abs(TM_evals[0]):
                TM_evals = TM_evals[::-1]
            tf_mine = time.time()
            print(f"    time taken = {tf_mine - t0_mine}")
            props["dominant TM eigenvalue"] = TM_evals[0]
            props["second-dominant TM eigenvalue magnitude"] = abs(TM_evals[1] / TM_evals[0])
            props["second-dominant TM eigenvalue phase"] = np.angle(TM_evals[1] / TM_evals[0])
            props["second-dominant TM eigenvalue real part"] = np.real(TM_evals[1] / TM_evals[0])
            props["second-dominant TM eigenvalue imag part"] = np.imag(TM_evals[1] / TM_evals[0])
            props["correlation length"] = 2 / np.log(abs(TM_evals[0]) / abs(TM_evals[1]))

        # Validate by calculating the transfer matrix spectrum directly (if the bond dimension is not too big to do this)
        if any([chi > 50 for chi in psi.chi]) or not check:
            # props['validation second-dominant TM singular value'] = np.nan
            # props['validation correlation length from svals'] = np.nan
            props["validation dominant TM eigenvalue"] = np.nan
            props["validation second-dominant TM eigenvalue magnitude"] = np.nan
            props["validation second-dominant TM eigenvalue phase"] = np.nan
            props["validation second-dominant TM eigenvalue real part"] = np.nan
            props["validation second-dominant TM eigenvalue imag part"] = np.nan
            props["validation correlation length"] = np.nan
            props["tenpy correlation length"] = np.nan
        elif not all([chi > 2 for chi in psi.chi]):
            # props['validation second-dominant TM singular value'] = np.nan
            # props['validation correlation length from svals'] = np.nan
            props["validation dominant TM eigenvalue"] = 1.0
            props["validation second-dominant TM eigenvalue magnitude"] = 0.0
            props["validation second-dominant TM eigenvalue phase"] = 0.0
            props["validation second-dominant TM eigenvalue real part"] = 0.0
            props["validation second-dominant TM eigenvalue imag part"] = 0.0
            props["validation correlation length"] = 0.0
            props["tenpy correlation length"] = 0.0
        else:
            AB = psi.get_theta(coarsegrain_from, n=coarsegrain_size, formL=1.0, formR=0.0)
            AB = copy.deepcopy(AB.to_ndarray())
            AB = rearrange(AB, "L ... R -> (...) L R")
            TM_dense = einsum(AB, np.conj(copy.deepcopy(AB)), "p l r, p lc rc -> l lc r rc")

            if check >= 2:
                I = np.eye(AB.shape[1])
                I_check = einsum(I, TM_dense, "l lc, l lc r rc -> r rc")
                assert np.allclose(
                    I_check, I
                ), f"The dominant left TM evec should be the identity, but running the identity through gave {I_check} {plt.imshow(abs(I_check))}"

            TM_dense = rearrange(TM_dense, "l lc r rc -> (r rc) (l lc)")

            val_svals = np.linalg.svd(TM_dense, full_matrices=False, compute_uv=False)

            props["validation second-dominant TM singular value"] = val_svals[1] / val_svals[0]
            props["validation correlation length from svals"] = 2 / np.log(
                val_svals[0] / val_svals[1]
            )

            # Validate the transfer matrix eigenvalues (only valid for infinite boundary conditions, and only if we're looking at the full unit cell)
            # Expand dimensions if TM_dense is not square
            if TM_dense.shape[0] != TM_dense.shape[1]:
                max_dim = max(TM_dense.shape)
                expanded = np.zeros((max_dim, max_dim))
                expanded[: TM_dense.shape[0], : TM_dense.shape[1]] = TM_dense
                TM_dense = expanded
                props["tenpy correlation length"] = np.nan
            else:
                if bc_MPS == "infinite":
                    props["tenpy correlation length"] = psi.correlation_length()
                else:
                    props["tenpy correlation length"] = np.nan
            val_evals = np.linalg.eigvals(TM_dense)
            props["validation dominant TM eigenvalue"] = val_evals[0]
            props["validation second-dominant TM eigenvalue magnitude"] = abs(
                val_evals[1] / val_evals[0]
            )
            props["validation second-dominant TM eigenvalue phase"] = np.angle(
                val_evals[1] / val_evals[0]
            )
            props["validation second-dominant TM eigenvalue real part"] = np.real(
                val_evals[1] / val_evals[0]
            )
            props["validation second-dominant TM eigenvalue imag part"] = np.imag(
                val_evals[1] / val_evals[0]
            )
            props["validation correlation length"] = 2 / np.log(
                abs(val_evals[0]) / abs(val_evals[1])
            )

    return props


def measure_tebd(tebd_engine, initial_energy=None, config_dict=None, **kwargs):
    """As above, but also store information about the curent energy, time, etc."""
    props = {}
    props["time"] = tebd_engine.evolved_time
    props["truncation error"] = tebd_engine.trunc_err.ov_err
    props.update(measure_psi(tebd_engine.psi, **kwargs))
    if initial_energy is not None:
        props["energy"] = np.mean(tebd_engine.model.bond_energies(tebd_engine.psi))
        props["energy difference"] = abs(props["energy"] - initial_energy) / initial_energy
    else:
        props["energy"] = 0
        props["energy difference"] = 0
    if config_dict is not None:
        props.update(config_dict)
    return props
