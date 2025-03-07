# %%
# Generate matrices on which to test simultaneous block diagonalization and simulateous block SVD algorithms
# These algorithms take a set of matrices $\{A_1 ... A_N\}$ and perform similarity transformations
# to yield a set of block diagonal matrices $\{B_1 ... B_n\}$ with the same finest block structure.
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
import tenpy
from opt_einops import (
    einsum,
    rearrange,
)  # Tensor opertations # ! pip install git+https://github.com/jordansauce/opt_einops
from scipy.stats import unitary_group  # Random unitaries
from tenpy.algorithms import tebd
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite

from instantaneous_benchmarks.correlation_vs_entanglement import run_random
from wavefunction_branching.measure import measure_psi
from wavefunction_branching.utils.tensors import make_json_serializable

###################################################
#############  ARTIFICICAL INPUT TENSORS  #########
###################################################


def block_sizes_to_rect(block_sizes) -> np.ndarray:
    """Converts from square block size format (a list of ints) to rectangular format (a np array formed from a list of 2-tuples)
    block_sizes: a list of ints (if the blocks are square), or a list of 2-tuples otherwise
    Output: numpy array of shape (len(block_sizes), 2), where output[i] = (left_dim_block[i], right_dim_block[i])"""
    return np.array([(x, x) if isinstance(x, int) else x for x in block_sizes])


def block_diagonal_matrices(block_sizes=[5, 5], N=4, noise_introduced: float = 0.0) -> np.ndarray:
    """Construct N random block-diagonal matrices with the specified block sizes
        Also introduce off-block noise if noise_introduced > 0
    Inputs:
        block_sizes: a list of ints (if the blocks are square), or a list of 2-tuples otherwise
    Output: numpy tensor of shape (N, sum(block_sizes), sum(block_sizes))"""
    block_sizes_ = block_sizes_to_rect(block_sizes)
    Ms = np.zeros((N, np.sum(block_sizes_[:, 0]), np.sum(block_sizes_[:, 1])), dtype=complex)
    Ms_ind_L = 0
    Ms_ind_R = 0
    for i in range(len(block_sizes_)):
        block = np.random.normal(
            size=(N, block_sizes_[i, 0], block_sizes_[i, 1])
        ) + 1.0j * np.random.normal(size=(N, block_sizes_[i, 0], block_sizes_[i, 1]))
        Ms[
            :,
            Ms_ind_L : (Ms_ind_L + block_sizes_[i, 0]),
            Ms_ind_R : (Ms_ind_R + block_sizes_[i, 1]),
        ] = block
        Ms_ind_L += block_sizes_[i, 0]
        Ms_ind_R += block_sizes_[i, 1]
    Ms += noise_introduced * (
        np.random.normal(size=Ms.shape) + 1.0j * np.random.normal(size=Ms.shape)
    )
    return Ms


def diagonal_dominant_matrices(
    D=50, N=4, γ=0.2, noise_introduced=0, diagonal_width=10
) -> np.ndarray:
    """Construct N random matrices which each have elements decaying away from the diagonal
        Also introduce off-diagonal noise if noise_introduced > 0
    Output: numpy tensor of shape (N, D, D)"""
    A = np.zeros((N, D, D), dtype=np.complex64)
    for i in range(N):
        for w in range(-diagonal_width, diagonal_width + 1):
            A[i] += (
                np.diag(np.random.randn(D - abs(w)), w)
                + 1.0j * np.diag(np.random.randn(D - abs(w)), w)
            ) * γ ** abs(w)
    A += noise_introduced * (np.random.random(A.shape) + 1.0j * np.random.random(A.shape))
    return A


def block_diagonal_matrices_exp_decaying_spectrum(
    block_sizes=[[30, 30], [30, 30]], N=4, decayrate=3.5, noise_introduced: float = 0.0
) -> np.ndarray:
    """Construct block diagonal matrices with an exponentially decaying spectrum"""
    block_sizes_ = block_sizes_to_rect(block_sizes)
    print(f"block_sizes_ = {block_sizes_}")
    A = np.zeros((N, np.sum(block_sizes_[:, 0]), np.sum(block_sizes_[:, 1])), dtype=complex)
    for n in range(N):
        ind_L = 0
        ind_R = 0
        for b in range(len(block_sizes_)):
            a = []
            minsize = min(block_sizes_[b, 0], block_sizes_[b, 1])
            for i in range(minsize):
                a += [np.exp(-np.random.random() * i * decayrate)]
            np.random.shuffle(a)
            D = np.zeros((block_sizes_[b, 0], block_sizes_[b, 1]), dtype=complex)
            D[:minsize, :minsize] += np.diag(a)
            A[n, ind_L : (ind_L + block_sizes_[b, 0]), ind_R : (ind_R + block_sizes_[b, 1])] = (
                unitary_group.rvs(block_sizes_[b, 0]) @ D @ unitary_group.rvs(block_sizes_[b, 1])
            )
            ind_L += block_sizes_[b, 0]
            ind_R += block_sizes_[b, 1]
    A += noise_introduced * np.random.normal(size=A.shape)
    return A


def blockdiag_from_spectrum_L(spectrum, N_matrices=2):
    """Construct block diagonal matrices with a left-spectrum provided as a list.
    Called by blockdiag_from_spectrum() (once for L, once for R)"""
    svals_in_block_a = spectrum[: len(spectrum) // 2]
    svals_in_block_b = spectrum[len(spectrum) // 2 :]

    U_a = unitary_group.rvs(len(svals_in_block_a))
    U_b = unitary_group.rvs(len(svals_in_block_b))
    Vh_a = unitary_group.rvs(len(svals_in_block_a) * N_matrices)[: len(svals_in_block_a), :]
    Vh_b = unitary_group.rvs(len(svals_in_block_b) * N_matrices)[: len(svals_in_block_b), :]

    block_a = einsum(U_a, svals_in_block_a, Vh_a, "l m, m, m R -> l R")
    block_b = einsum(U_b, svals_in_block_b, Vh_b, "l m, m, m R -> l R")
    block_a = rearrange(block_a, "l (n r) ->  n l r", n=N_matrices)
    block_b = rearrange(block_b, "l (n r) ->  n l r", n=N_matrices)

    blockdiag = np.zeros((N_matrices, len(spectrum), len(spectrum)), dtype=complex)
    blockdiag[:, : len(spectrum) // 2, : len(spectrum) // 2] = block_a
    blockdiag[:, len(spectrum) // 2 :, len(spectrum) // 2 :] = block_b

    return blockdiag


def blockdiag_from_spectrum(spectrum, N_matrices=4, noise_introduced=0.0):
    """Construct block diagonal matrices with a spectrum provided as a list.
    This will be the left spectrum and the right spectrum.
    This is done by generating two blockdiagonal matrices: one with the specturm on the left,
    and the other with the spectrum on the right."""
    spectrum = np.random.permutation(spectrum) ** 0.6
    L = blockdiag_from_spectrum_L(spectrum, N_matrices=N_matrices // 2)
    R = blockdiag_from_spectrum_L(spectrum, N_matrices=N_matrices - N_matrices // 2)

    if noise_introduced > 0.0:
        # Prevent perfect block diagonality while maintaining the spectrum,
        # by inserting a weak unitary mixing matrix between L and R
        H = np.random.randn(L.shape[-1], L.shape[-1]) + 1.0j * np.random.randn(
            L.shape[-1], L.shape[-1]
        )
        H += np.conj(H.T)
        U_noise = scipy.linalg.expm(noise_introduced * H)
        L = einsum(L, U_noise, "n l r, r m -> n l m")

    blockdiag = einsum(L, R, "n1 l m, n2 r m -> n1 n2 l r")
    blockdiag = rearrange(blockdiag, "n1 n2 l r -> (n1 n2) l r")

    return blockdiag


###################################################
#############  REAL MPS INPUT TENSORS  ############
###################################################


# Redefine tenpy's TFIModel to make it consistent with "Converting long-range entanglement into mixture" by swapping Sigmax and Sigmaz
class TFIModel(CouplingMPOModel):
    r"""Transverse field Ising model on a general lattice.

    The Hamiltonian reads:

    .. math ::
        H = - \sum_{\langle i,j\rangle, i < j} \mathtt{J} \sigma^z_i \sigma^z_{j}
            - \sum_{i} \mathtt{g} \sigma^x_i

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs, each pair appearing
    exactly once.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`TFIModel` below.

    Options
    -------
    .. cfg:config :: TFIModel
        :include: CouplingMPOModel

        conserve : None | 'parity'
            What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
        sort_charge : bool | None
            Whether to sort by charges of physical legs.
            See change comment in :class:`~tenpy.networks.site.Site`.
        J, g : float | array
            Coupling as defined for the Hamiltonian above.

    """

    def init_sites(self, model_params):
        conserve = model_params.get("conserve", "parity")
        assert conserve != "Sz"
        if conserve == "best":
            conserve = "parity"
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        sort_charge = model_params.get("sort_charge", None)
        site = SpinHalfSite(conserve=conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):
        J = np.asarray(model_params.get("J", 1.0))
        g = np.asarray(model_params.get("g", 1.0))
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g, u, "Sigmax")
        for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:
            self.add_coupling(-J, u1, "Sigmaz", u2, "Sigmaz", dx)
        # done


class TFIChain(TFIModel, NearestNeighborModel):
    """The :class:`TFIModel` on a Chain, suitable for TEBD.

    See the :class:`TFIModel` for the documentation of parameters.
    """

    default_lattice = Chain
    force_default_lattice = True


def save_matrices(
    saves_dict: dict,
    Bs: np.ndarray | None = None,
    As: np.ndarray | None = None,
    svals_L: np.ndarray | None = None,
    svals_R: np.ndarray | None = None,
    scramble_kind: str = "null",
    folder: str | Path = "block_diagonal_test_data",
    subfolder: str = "uncategorized",
    block_sizes: str | list[int] = "null",
    N: str | int = "null",
    noise: str | float = "null",
    gamma: str | float = "null",
    decayrate: str | float = "null",
    t: str | float = "null",
    dt: str | float = "null",
    J: str | float = "null",
    g: str | float = "null",
    L: str | int = "null",
    chi_max: str | int = "null",
    svd_min: str | float = "null",
    form: str | tuple[float, float] = "null",
    bell_like: bool = False,
    square_blocks: bool = True,
    equal_sized_blocks: bool = True,
    from_frias_perez: bool = False,
    directory_name: str = "directory",
):
    if Bs is not None and scramble_kind != "null":
        X, Y = generate_scrambling_matrices(scramble_kind, dim_L=Bs.shape[1], dim_R=Bs.shape[2])
        As = X @ Bs @ Y

    if As is None and Bs is not None:
        As = Bs
    if Bs is None and As is not None:
        Bs = As

    saves_dict["kind"].append(subfolder)
    saves_dict["scramble_kind"].append(scramble_kind)
    saves_dict["N_matrices"].append("null" if N == "null" else int(N))
    saves_dict["block_sizes"].append(block_sizes)
    saves_dict["dim_L"].append("null" if As is None else As.shape[1])
    saves_dict["dim_R"].append("null" if As is None else As.shape[2])
    saves_dict["noise_introduced"].append(noise)
    saves_dict["gamma"].append(gamma)
    saves_dict["decayrate"].append(decayrate)
    saves_dict["t"].append(t)
    saves_dict["dt"].append(dt)
    saves_dict["J"].append(J)
    saves_dict["g"].append(g)
    saves_dict["L"].append(L)
    saves_dict["chi_max"].append(chi_max)
    saves_dict["svd_min"].append(svd_min)
    saves_dict["form"].append(form)
    saves_dict["form_str"].append(str(form))
    form_L = "null" if form == "null" else float(form[0])
    form_R = "null" if form == "null" else float(form[1])
    saves_dict["form_L"].append(form_L)
    saves_dict["form_R"].append(form_R)
    timestamp = str(datetime.now()).replace(" ", "_")[:-4]
    saves_dict["bell_like"].append(bell_like)
    saves_dict["square_blocks"].append(square_blocks)
    saves_dict["equal_sized_blocks"].append(equal_sized_blocks)
    saves_dict["timestamp"].append(timestamp)
    saves_dict["from_frias_perez"].append(from_frias_perez)

    if t != "null":
        save_str = str(
            (
                Path(folder) / f"{subfolder}/g-{g}_J-{J}_t-{t}_form-L{form_L}-R{form_R}_{timestamp}"
            ).relative_to(Path(folder).parent)
        )
    else:
        save_str = str(
            (
                Path(folder)
                / f"{subfolder}/{scramble_kind}/block_sizes-{block_sizes}_N-{N}_noise-{noise}_{timestamp}"
            ).relative_to(Path(folder).parent)
        )

    save_str = (
        save_str.replace(" ", "")
        .replace(",", "-")
        .replace("[", "")
        .replace("]", "")
        .replace(":", "-")
        .replace(".", "-")
    )

    if Bs is None:
        saves_dict["Bs_file"].append("null")
    else:
        Bs_path = Path(folder).parent / Path(save_str + "_Bs.npy")
        Bs_path.parent.mkdir(parents=True, exist_ok=True)
        saves_dict["Bs_file"].append(str(Bs_path))
        with open(Bs_path.absolute(), "wb") as f:
            np.save(f, Bs)
        print(f"Bs_path: {Bs_path.absolute()}")

    if As is None:
        if Bs is None:
            saves_dict["As_file"].append("null")
        else:
            saves_dict["As_file"].append(saves_dict["Bs_file"][-1])
    else:
        As_path = Path(folder).parent / Path(save_str + "_As.npy")
        As_path.parent.mkdir(parents=True, exist_ok=True)
        saves_dict["As_file"].append(str(As_path))
        with open(As_path.absolute(), "wb") as f:
            np.save(f, As)
        print(f"As_path: {As_path.absolute()}")

    if svals_L is None:
        saves_dict["svals_L_file"].append("null")
    else:
        svals_L_path = Path(folder).parent / Path(save_str + "_svals_L.npy")
        svals_L_path.parent.mkdir(parents=True, exist_ok=True)
        saves_dict["svals_L_file"].append(str(svals_L_path))
        with open(svals_L_path.absolute(), "wb") as f:
            np.save(f, svals_L)
        print(f"svals_L_path: {svals_L_path.absolute()}")

    if svals_R is None:
        saves_dict["svals_R_file"].append("null")
    else:
        svals_R_path = Path(folder).parent / Path(save_str + "_svals_R.npy")
        svals_R_path.parent.mkdir(parents=True, exist_ok=True)
        saves_dict["svals_R_file"].append(str(svals_R_path))
        with open(svals_R_path.absolute(), "wb") as f:
            np.save(f, svals_R)
        print(f"svals_R_path: {svals_R_path.absolute()}")

    saves_dict["save_str"].append(save_str)

    print(f"save_str: {save_str}")

    with open(Path(folder) / f"{directory_name}.json", "w", encoding="utf-8") as f:
        json.dump(saves_dict, f, ensure_ascii=False, indent=4)
    print(f"Saved json to {folder}/{directory_name}.json")
    # if np.random.random() < 0.1:
    #     vis(Bs, 'Bs '+save_str)
    return saves_dict


def run_TEBD_ising(
    psi: MPS | None = None,
    model: TFIChain | None = None,
    chi=200,  # bond dimension
    chi_terminate_at=None,
    delta_t=0.005,  # timestep
    t_evo=3.5,  # total time to evolve for
    steps_per_output=50,
    N_sites=2,  # number of sites in unit cell
    svd_min=1e-9,  # singular value cutoff
    trunc_cut=1e-5,
    start_time=0,
    J=1.0,  # ising model σzσz term strength
    g=2.0,  # ising model σx term strength
    g0=np.inf,  # ising model σx term strength before the quench at t=0
    save_prefix="natural_AB",
    folder: str | Path = "block_diagonal_test_data",
    save_forms=[(0.5, 0.5), (1.0, 1.0)],
    saves_dict=None,
):
    # ======== ========== ==========================================================================
    # `form`   tuple      description
    # ======== ========== ==========================================================================
    # ``'B'``  (0, 1)     right canonical: ``_B[i] = -- Gamma[i] -- s[i+1]--``
    #                     The default form, which algorithms assume.
    # ``'C'``  (0.5, 0.5) symmetric form: ``_B[i] = -- s[i]**0.5 -- Gamma[i] -- s[i+1]**0.5--``
    # ``'A'``  (1, 0)     left canonical: ``_B[i] = -- s[i] -- Gamma[i] --``.
    # ``'G'``  (0, 0)     Save only ``_B[i] = -- Gamma[i] --``.
    # ``'Th'`` (1, 1)     Form of a local wave function `theta` with singular value on both sides.
    #                     ``psi.get_B(i, 'Th') is equivalent to ``psi.get_theta(i, n=1)``.
    # ``None`` ``None``   General non-canonical form.
    #                     Valid form for initialization, but you need to call
    #                     :meth:`~tenpy.networks.mps.MPS.canonical_form` (or similar)
    #                     before using algorithms.
    # ======== ========== ==========================================================================
    if saves_dict is None:
        saves_dict = defaultdict(list)
    if model is None:
        model = TFIChain(
            {
                "L": N_sites,
                "J": J,
                "g": g,
                "bc_MPS": "infinite",
                "conserve": None,
                "sort_charge": False,
            }
        )
    sites = model.lat.mps_sites()
    if psi is None:
        psi = MPS.from_product_state(
            sites, [[1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]] * N_sites, "infinite"
        )  # start in an eigenstate of sigma_x (all spins pointing right)
        assert psi is not None
        if g0 != np.inf:
            # Perform DMRG to get the ground state of the Hamiltonian with g=g0
            pre_quench_model = TFIChain(
                {
                    "L": N_sites,
                    "J": J,
                    "g": g0,
                    "bc_MPS": "infinite",
                    "conserve": None,
                    "sort_charge": False,
                }
            )
            dmrg_params = {
                "mixer": None,  # setting this to True helps to escape local minima
                "max_E_err": 1.0e-8,
                "trunc_params": {
                    "chi_max": 30,
                    "svd_min": svd_min,
                    "trunc_cut": trunc_cut,
                },
            }
            eng = tenpy.algorithms.dmrg.TwoSiteDMRGEngine(psi, pre_quench_model, dmrg_params)
            E, psi = (
                eng.run()
            )  # equivalent to dmrg.run() up to the return parameters. #type: ignore
            print(f"initial energy after DMRG = {E:.13f}")
            assert psi is not None
            print("initial bond dimensions after DMRG: ", psi.chi)
            mag_x = np.mean(psi.expectation_value("Sigmax"))  # type: ignore
            mag_z = np.mean(psi.expectation_value("Sigmaz"))  # type: ignore
            print(f"initial <sigma_x> after DMRG = {mag_x:.5f}")
            print(f"initial <sigma_z> after DMRG = {mag_z:.5f}")
            print("initial correlation length after DMRG:", psi.correlation_length())
    assert model is not None
    assert psi is not None
    tebd_params = {
        "start_time": start_time,
        "order": 2,
        "dt": delta_t,
        "N_steps": steps_per_output,
        # 'max_error_E': 1.e-8,
        "trunc_params": {"chi_max": chi, "svd_min": svd_min, "trunc_cut": trunc_cut},
    }
    tebd_engine = tebd.TEBDEngine(psi, model, tebd_params)
    psis = []
    for i in range(int(np.ceil(t_evo / (delta_t * steps_per_output)))):
        tebd_engine.run()

        props = measure_psi(tebd_engine.psi, check=False)
        for form in save_forms:
            As = rearrange(
                psi.get_theta(0, n=2, formL=form[0], formR=form[1]).to_ndarray(),
                "L p1 p2 R -> (p1 p2) L R",
            )
            svals_L = tenpy.tools.misc.to_array(psi.get_SL(0))
            svals_R = svals_L
            for key in props:
                saves_dict[key].append(make_json_serializable(props[key]))
            saves_dict = save_matrices(
                saves_dict,
                As=As,
                svals_L=svals_L,
                svals_R=svals_R,
                N=As.shape[0],
                folder=folder,
                subfolder=f"ising_evo_g={g:.2f}_g0={g0:.2f}_J={J:.2f}",
                t=round(tebd_engine.evolved_time, ndigits=5),
                dt=delta_t,
                J=J,
                g=g,
                L=N_sites,
                chi_max=chi,
                svd_min=svd_min,
                form=form,
                directory_name="directory-ising-evo",
            )
        if chi_terminate_at is not None and np.mean(psi.chi) > chi_terminate_at:
            break
    return saves_dict


def run_TEBD_random_evo(
    model_name,
    saves_dict=None,
    chi_max=100,  # bond dimension
    svd_min=1e-9,
    delta_t=0.005,  # timestep
    t_evo=5.0,  # total time to evolve for
    steps_per_output=100,
    N_sites=2,
    folder: str | Path = "block_diagonal_test_data",
    **kwargs,
):
    if saves_dict is None:
        saves_dict = defaultdict(list)
    form = (1.0, 1.0)
    psi, model, properties_over_time = run_random(
        model_name,
        chi_max=chi_max,
        delta_t=delta_t,
        t_evo=t_evo,
        steps_per_output=steps_per_output,
        **kwargs,
    )
    As = rearrange(
        psi.get_theta(0, n=2, formL=form[0], formR=form[1]).to_ndarray(), "L p1 p2 R -> (p1 p2) L R"
    )
    svals_L = tenpy.tools.misc.to_array(psi.get_SL(0))
    svals_R = svals_L
    props = measure_psi(psi, check=False)
    print("props:")
    print(props)
    for key in props:
        saves_dict[key].append(make_json_serializable(props[key]))
    saves_dict = save_matrices(
        saves_dict,
        As=As,
        svals_L=svals_L,
        svals_R=svals_R,
        N=As.shape[0],
        folder=folder,
        subfolder=model_name,
        t=t_evo,
        dt=delta_t,
        L=2,
        chi_max=chi_max,
        svd_min=svd_min,
        form=form,
    )
    return saves_dict


###################################################
##  SAVE A TEST SUITE OF INPUT TENSORS TO DISK  ##
###################################################


def generate_scrambling_matrices(
    scramble_kind, dim_L: int, dim_R: int
) -> tuple[np.ndarray, np.ndarray]:
    assert scramble_kind in ["UU", "UV", "XX", "XY", "null", None], (
        "scramble_kind must be one of ['UU', 'UV', 'XX', 'XY' 'null' or None]"
    )
    if scramble_kind is None or scramble_kind == "null":
        return np.eye(dim_L), np.eye(dim_R)
    if scramble_kind == "UU" or scramble_kind == "XX":
        assert dim_L == dim_R, (
            "Dimension mismatch: UU and XX scrambling require square matrices (dim_L = dim_R)"
        )
    if scramble_kind == "UU":
        U = unitary_group.rvs(dim_L)
        return U, U.conj().T
    if scramble_kind == "UV":
        U = unitary_group.rvs(dim_L)
        V = unitary_group.rvs(dim_R)
        return U, V
    if scramble_kind == "XX":
        X = np.random.random(size=(dim_L, dim_L))
        return X, np.linalg.pinv(X)
    if scramble_kind == "XY":
        X = np.random.random(size=(dim_L, dim_L))
        Y = np.random.random(size=(dim_R, dim_R))
        return X, Y
    assert False, f"Invalid scramble_kind: {scramble_kind}"


if __name__ == "__main__":
    current_path = Path(__file__).parent.absolute()
    folder = current_path / "block_diagonal_test_data"
    folder.mkdir(parents=True, exist_ok=True)

    print("\n\n\n\n\n\n--------------------------------")
    print("Running ising quench evolutions")
    print("--------------------------------\n\n")
    saves_dict_ising = defaultdict(list)
    for chi_max in [50, 100, 200, 400]:
        print(f"Running ising evo with chi_max = {chi_max}")
        saves_dict_ising = run_TEBD_ising(
            psi=None,
            model=None,
            chi=chi_max,  # bond dimension
            delta_t=0.01,  # timestep
            t_evo=8.0,  # total time to evolve for
            steps_per_output=50,
            N_sites=2,  # number of sites in unit cell
            svd_min=1e-9,  # singular value cutoff
            trunc_cut=5e-7,  # singular value cutoff
            start_time=0,
            J=1.0,  # ising model σzσz term strength
            g=2.0,  # ising model σx term strength
            folder=folder,
            save_forms=[(1.0, 1.0)],
            saves_dict=saves_dict_ising,
        )
    print(saves_dict_ising)

    print("\n\n\n\n\n\n--------------------------------")
    print("Running random quench evolutions")
    print("--------------------------------\n\n")
    saves_dict_random = defaultdict(list)
    for chi_max in [30, 50, 100, 150]:
        for i in range(20):
            for model_name in ["tf_ising", "spins", "aklt"]:
                print(f"Running {model_name}, chi_max = {chi_max}")
                saves_dict_random = run_TEBD_random_evo(
                    model_name, saves_dict=saves_dict_random, folder=folder, chi_max=chi_max
                )
                print(saves_dict_random)

# %%
