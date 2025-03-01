"""Hamiltonians and different types of evolution setups (eg. random circuits) for TEBD"""

import numpy as np
import tenpy
import tenpy.linalg.np_conserved as npc
from numpy.random import default_rng
from opt_einops import (
    rearrange,
)  # Tensor opertations # ! pip install git+https://github.com/jordansauce/opt_einops
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.networks.site import SpinHalfSite

rng = default_rng()


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
        self.J = J.item()
        self.g = g.item()


class TFIChain(TFIModel, NearestNeighborModel):
    """The :class:`TFIModel` on a Chain, suitable for TEBD.

    See the :class:`TFIModel` for the documentation of parameters.
    """

    default_lattice = Chain
    force_default_lattice = True


# Random local_dim^2 permutation with a Hadamard on one of the qubits
def noah_random_gate(dimension=2, num_hadamards=1, hadamard_rate=1.0):
    gate = np.zeros((dimension * dimension, dimension * dimension), np.cdouble)
    remaining_indices = np.arange(dimension * dimension)

    r = np.random.random()
    print(f"    hadamard_rate = {hadamard_rate}")
    if r <= hadamard_rate:
        print("    hadamard inserted")
        hadamard_indices = rng.choice(dimension * dimension, 2 * num_hadamards, replace=False)
        for j in range(num_hadamards):
            gate[hadamard_indices[2 * j], hadamard_indices[2 * j]] = 1.0 / np.sqrt(2.0)
            gate[hadamard_indices[2 * j], hadamard_indices[2 * j + 1]] = 1.0 / np.sqrt(2.0)
            gate[hadamard_indices[2 * j + 1], hadamard_indices[2 * j]] = -1.0 / np.sqrt(2.0)
            gate[hadamard_indices[2 * j + 1], hadamard_indices[2 * j + 1]] = 1.0 / np.sqrt(2.0)
        remaining_indices = np.delete(remaining_indices, hadamard_indices)

    perm = rng.permutation(remaining_indices)
    for i in range(len(perm)):
        gate[remaining_indices[i], perm[i]] = 1.0

    # assert np.allclose(gate @ gate.conj().T, np.eye(gate.shape[0])), "gate is not unitary"
    # assert np.allclose(gate.conj().T @ gate, np.eye(gate.shape[1])), "gate is not unitary"
    return gate


class NoahEvolution(tenpy.algorithms.tebd.RandomUnitaryEvolution):
    def __init__(self, psi, hadamard_rate, options, **kwargs):
        self.hadamard_rate = hadamard_rate
        super().__init__(psi, options, **kwargs)

    def calc_U(self):
        """Draw new random two-site unitaries replacing the usual `U` of TEBD.

        .. cfg:configoptions :: RandomUnitaryEvolution

            distribution_func : str | function
                Function or name for one of the matrix ensembles in
                :mod:`~tenpy.linalg.random_matrix` which generates unitaries (or a subset of them).
                To be used as `func` for generating unitaries with
                :meth:`~tenpy.linalg.np_conserved.Array.from_func_square`, i.e. the `U` still
                preserves the charge block structure!
            distribution_func_kwargs : dict
                Extra keyword arguments for `distribution_func`.
        """
        # func = noah_random_gate
        sites = self.psi.sites
        L = len(sites)
        U_bonds = []
        for i in range(L):
            if i == 0 and self.psi.finite:
                U_bonds.append(None)
            else:
                gate = noah_random_gate(hadamard_rate=self.hadamard_rate)
                npc_gate = npc.Array.from_ndarray_trivial(
                    rearrange(gate, "(pa pb) (pac pbc) -> pa pb pac pbc", pa=2, pac=2),
                    labels=["p0", "p1", "p0*", "p1*"],
                )
                U_bonds.append(npc_gate)
        self._U = [U_bonds]


# Random local_dim^2 permutation with a Hadamard on one of the qubits
def h_cnot_random_gate(hadamard_rate=0.5, cnot_rate=0.5, check=False):  # -> Any:
    gate = np.eye(4, dtype=complex)

    # Random chance of a CNOT gate
    rc = np.random.random()
    if rc < cnot_rate:
        cnot_gate = np.zeros((4, 4), dtype=complex)
        control_qubit = np.random.choice([0, 1])
        if control_qubit == 0:
            cnot_gate[0, 0] = 1.0
            cnot_gate[1, 1] = 1.0
            cnot_gate[2, 3] = 1.0
            cnot_gate[3, 2] = 1.0
        else:
            cnot_gate[0, 1] = 1.0
            cnot_gate[1, 0] = 1.0
            cnot_gate[2, 2] = 1.0
            cnot_gate[3, 3] = 1.0
        gate = gate @ cnot_gate

    # Random chance of a Hadamard gates
    rh = np.random.random()
    if rh < hadamard_rate:
        H_gate = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
        hadamard_qubit = np.random.choice([0, 1])
        if hadamard_qubit == 0:
            gate = gate @ np.kron(H_gate, np.eye(2))
        else:
            gate = gate @ np.kron(np.eye(2), H_gate)

    # Randomize the order of the gates
    if np.random.choice([True, False]):
        gate = np.conj(np.transpose(gate))

    if check:
        assert np.allclose(gate @ gate.conj().T, np.eye(4)), "gate is not unitary"
        assert np.allclose(gate.conj().T @ gate, np.eye(4)), "gate is not unitary"
    return gate


# Random clifford gate
def clifford_random_gate(hadamard_rate=0.5, cnot_rate=0.5, s_rate=0.5, check=False):
    gate = h_cnot_random_gate(hadamard_rate=hadamard_rate, cnot_rate=cnot_rate)

    # Random chance of an S gate
    r = np.random.random()
    if r < s_rate:
        S_gate = np.array([[1.0, 0.0], [0.0, 1.0j]])
        S_qubit = np.random.choice([0, 1])
        if S_qubit == 0:
            gate = gate @ np.kron(S_gate, np.eye(2))
        else:
            gate = gate @ np.kron(np.eye(2), S_gate)

    # Randomize the order of the gates
    if np.random.choice([True, False]):
        gate = np.transpose(gate)
    if np.random.choice([True, False]):
        gate = np.conj(gate)

    if check:
        assert np.allclose(gate @ gate.conj().T, np.eye(4)), "gate is not unitary"
        assert np.allclose(gate.conj().T @ gate, np.eye(4)), "gate is not unitary"
    return gate


class CliffordEvolution(tenpy.algorithms.tebd.RandomUnitaryEvolution):
    def __init__(self, psi, hadamard_rate, cnot_rate, s_rate, options, **kwargs):
        self.hadamard_rate = hadamard_rate
        self.cnot_rate = cnot_rate
        self.s_rate = s_rate
        super().__init__(psi, options, **kwargs)

    def calc_U(self):
        """Draw new random two-site unitaries replacing the usual `U` of TEBD.

        .. cfg:configoptions :: RandomUnitaryEvolution

            distribution_func : str | function
                Function or name for one of the matrix ensembles in
                :mod:`~tenpy.linalg.random_matrix` which generates unitaries (or a subset of them).
                To be used as `func` for generating unitaries with
                :meth:`~tenpy.linalg.np_conserved.Array.from_func_square`, i.e. the `U` still
                preserves the charge block structure!
            distribution_func_kwargs : dict
                Extra keyword arguments for `distribution_func`.
        """
        # func = noah_random_gate
        sites = self.psi.sites
        L = len(sites)
        U_bonds = []
        for i in range(L):
            if i == 0 and self.psi.finite:
                U_bonds.append(None)
            else:
                gate = clifford_random_gate(
                    hadamard_rate=self.hadamard_rate, cnot_rate=self.cnot_rate, s_rate=self.s_rate
                )
                npc_gate = npc.Array.from_ndarray_trivial(
                    rearrange(gate, "(pa pb) (pac pbc) -> pa pb pac pbc", pa=2, pac=2),
                    labels=["p0", "p1", "p0*", "p1*"],
                )
                U_bonds.append(npc_gate)
        self._U = [U_bonds]


MODELS = {
    "noah": lambda x: "noah",  # Noah's model of random permutations and Hadamards
    "clifford": lambda x: "clifford",  # Noah's model of random permutations and Hadamards
    "tf_ising": TFIChain,  # Prototypical example of a quantum model: the transverse field Ising model.
    # 'xxz_chain': tenpy.models.XXZChain, # Prototypical example of a 1D quantum model: the spin-1/2 XXZ chain.
    "spins": tenpy.models.spins.SpinChain,  # Nearest-neighbor spin-S models.
    "spins_nnn": tenpy.models.spins_nnn.SpinChainNNN,  # Next-Nearest-neighbor spin-S models.
    # 'fermions_spinless': tenpy.models.fermions_spinless.FermionModel, # Spinless fermions with hopping and interaction.
    "hubbard": tenpy.models.hubbard.BoseHubbardChain,  # Bosonic and fermionic Hubbard models.
    # 'tj_model': tenpy.models.tj_model.tJModel, # tJ model
    "aklt": tenpy.models.aklt.AKLTChain,  # Prototypical example of a 1D quantum model constructed from bond terms: the AKLT chain.
    # 'hofstadter_bosons': tenpy.models.hofstadter.HofstadterBosons, # Cold atomic (Harper-)Hofstadter model on a strip or cylinder.
    # 'hofstadter_fermions': tenpy.models.hofstadter.HofstadterFermions, # Cold atomic (Harper-)Hofstadter model on a strip or cylinder.
    # 'haldane_bosons': tenpy.models.BosonicHaldaneModel, # Bosonic Haldane model.
    # 'haldane_fermions': tenpy.models.FermionicHaldaneModel, # fermionic Haldane model.
    # 'toric_code': tenpy.models.toric_code.ToricCode, # Kitaev's exactly solvable toric code model.
    # 'clock': tenpy.models.clock.ClockModel, # Quantum Clock model.
    "random_circuit": lambda x: "random_circuit",
}
