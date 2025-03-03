# %%
import glob
import pickle
import signal
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
# from tenpy.models.lattice import Chain
# from tenpy.networks.site import SpinHalfSite
# from scipy.stats import unitary_group
import seaborn as sns
import tenpy
import tenpy.linalg.np_conserved as npc
from numpy.random import default_rng
from opt_einops import rearrange

# from tenpy.networks.mps import MPS
from tenpy.algorithms import tebd

rng = default_rng()


BC_MPS = "infinite"  #
N_SITES = 2
COARSEGRAIN_SIZE = 2
# N_SITES = 10
# COARSEGRAIN_SIZE = 3
COARSEGRAIN_SIZE = min(N_SITES, COARSEGRAIN_SIZE)
COARSEGRAIN_FROM = max(0, int(N_SITES / 2 - COARSEGRAIN_SIZE / 2))
NOW = {datetime.now().strftime("%Y-%m-%d")}
# NOW = '2024-05-23 - Copy'
CURRENT_PATH = Path(__file__).parent.absolute()
PICKLE_DIR = CURRENT_PATH / f"pickles/{NOW}"
PICKLE_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR = CURRENT_PATH / f"figures/{NOW}"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


from wavefunction_branching.measure import measure_psi


# Random local_dim^2 permutation with a Hadamard on one of the qubits
def noah_random_gate(dimension=2, num_hadamards=1, hadamard_rate=1.0):
    gate = np.zeros((dimension * dimension, dimension * dimension), np.cdouble)
    remaining_indices = np.arange(dimension * dimension)

    r = rng.random()
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


def measure_tebd(tebd_engine, initial_energy=None, config_dict=None):
    """As above, but also store information about the curent energy, time, etc."""
    props = {}
    props["time"] = tebd_engine.evolved_time
    props["truncation error"] = tebd_engine.trunc_err.ov_err
    props.update(measure_psi(tebd_engine.psi))
    if initial_energy is not None:
        props["energy"] = np.mean(tebd_engine.model.bond_energies(tebd_engine.psi))
        props["energy difference"] = abs(props["energy"] - initial_energy) / initial_energy
    else:
        props["energy"] = 0
        props["energy difference"] = 0
    if config_dict is not None:
        props.update(config_dict)
    return props


def run_TEBD(
    psi: tenpy.networks.mps.MPS,  # initial state
    model: tenpy.models.model.Model | str,  # Hamiltonian (Ising model)
    chi_max=256,  # maximum bond dimension
    terminate_at_chi_max=True,  # terminate when bond dimension reaches chi_max
    delta_t=0.01,  # timestep
    t_evo=50,  # total time to evolve for
    steps_per_output=10,
    svd_min=1e-8,  # singular value cutoff
    trunc_cut=5e-4,  # max singular value discarded sum
    max_total_trunc_err=1.0,
    terminate_at_max_total_trunc_err=True,  # terminate when bond dimension reaches chi_max
    start_time=0,
    hadamard_rate=None,
    cnot_rate=None,
    s_rate=None,
    config_dict=None,
) -> tuple[tenpy.networks.mps.MPS, tenpy.models.model.Model | str, pd.DataFrame]:
    """Perform iTEBD time evolution, starting from some TeNPy MPS state psi, evolving under the
    Transverse Field Ising Chain Hamiltonian in `model` with σzσz coupling strength = J and σx
    field strength = g for a total time of t_evo (natural units)"""
    # current_path = Path(__file__).parent.absolute()
    # folder = current_path / '{NOW}/tensors'
    if config_dict is None:
        config_dict = {}

    tebd_params = {
        "start_time": start_time,
        "order": 2,
        "dt": delta_t,
        "N_steps": steps_per_output,
        # 'max_error_E': 1.e-8,
        "trunc_params": {
            "chi_max": chi_max,
            "svd_min": svd_min,
            "trunc_cut": trunc_cut,
        },
    }
    if isinstance(model, str):
        delta_t = 1
        steps_per_output = 1
        tebd_params["dt"] = 1
        tebd_params["N_steps"] = 1
        initial_energy = None
        if model == "noah":
            if hadamard_rate is None:
                hadamard_rate = config_dict["hadamard_rate"]
            print(f"Hadamard rate = {hadamard_rate}")
            tebd_engine = NoahEvolution(psi, hadamard_rate=hadamard_rate, options=tebd_params)
        elif model == "clifford":
            if hadamard_rate is None:
                hadamard_rate = config_dict["hadamard_rate"]
            if cnot_rate is None:
                cnot_rate = config_dict["cnot_rate"]
            if s_rate is None:
                s_rate = config_dict["s_rate"]
            print(f"Hadamard rate = {hadamard_rate}, CNOT rate = {cnot_rate}, S rate = {s_rate}")
            tebd_engine = CliffordEvolution(
                psi,
                hadamard_rate=hadamard_rate,
                cnot_rate=cnot_rate,
                s_rate=s_rate,
                options=tebd_params,
            )
        else:
            tebd_engine = tenpy.algorithms.tebd.RandomUnitaryEvolution(psi, tebd_params)
    else:
        tebd_engine = tebd.TEBDEngine(psi, model, tebd_params)
        try:
            initial_energy = np.mean(tebd_engine.model.bond_energies(tebd_engine.psi))
        except:
            initial_energy = None

    properties_over_time = pd.DataFrame(columns=["time"])
    properties_over_time = pd.concat(
        [properties_over_time, pd.DataFrame([measure_tebd(tebd_engine, config_dict=config_dict)])],
        ignore_index=True,
    )

    save_tensor_at_chis = [25, 50, 100, 150]
    saves_dict = {}

    for i in range(int(np.ceil(t_evo / (delta_t * steps_per_output)))):
        tebd_engine.run()
        props = measure_tebd(tebd_engine, initial_energy=initial_energy, config_dict=config_dict)
        properties_over_time = pd.concat(
            [properties_over_time, pd.DataFrame([props])], ignore_index=True
        )
        print(props)
        if terminate_at_chi_max and max(psi.chi) >= chi_max:
            print(f"Bond dimension of {max(psi.chi)} exceeds chi_max of {chi_max}: terminating ")
            break
        if terminate_at_max_total_trunc_err and tebd_engine.trunc_err.ov_err >= max_total_trunc_err:
            print(
                f"Total truncation error of {tebd_engine.trunc_err.ov_err} exceeds max_total_trunc_err of {max_total_trunc_err}: terminating "
            )
            break
        # if len(save_tensor_at_chis) > 0:
        #     if props['average bond dimension'] > save_tensor_at_chis[0]:
        #         chis_passed = {save_tensor_at_chis}
        #         if props['correlation_length'] > 5.0:
        #             form = [1.0,1.0]
        #             As = rearrange(
        #                 psi.get_theta(0,n=2, formL=form[0], formR=form[1]).to_ndarray(),
        #                 'L p1 p2 R -> (p1 p2) L R'
        #             )
        #             svals_L = tenpy.tools.misc.to_array(psi.get_SL(0))
        #             svals_R = svals_L
        #             saves_dict = save_matrices(
        #                 saves_dict,
        #                 As = As,
        #                 svals_L = svals_L,
        #                 svals_R = svals_R,
        #                 N = As.shape[0],
        #                 folder=folder,
        #                 subfolder = 'ising_evo',
        #                 t = round(tebd_engine.evolved_time, ndigits=5),
        #                 dt = delta_t,
        #                 svd_min = svd_min,
        #                 form = form,
        #             )
    return psi, model, properties_over_time


##############################^^^ run_TEBD() ^^^##############################################


def plot_properties(
    properties,
    ys: str | list[str] = "correlation length",
    xs: str | list[str] = "entanglement entropy",
):
    plt.rcParams["figure.figsize"] = (15, 8)
    plt.figure(dpi=800)
    if isinstance(ys, str):
        ys = [ys]
    if isinstance(xs, str):
        xs = [xs]
    for y in ys:
        for x in xs:
            sns.lineplot(
                properties, y=y, x=x, hue="model", style="trial number", legend=False, alpha=0.35
            )
            sns.scatterplot(properties, y=y, x=x, hue="model", alpha=0.0)
            sns.scatterplot(properties, y=y, x=x, hue="model", style="trial number", legend=False)
    plt.savefig(
        PLOTS_DIR
        / (
            (
                ys[0]
                + (f"plus_{len(ys)}_more" if len(ys) > 1 else "")
                + " vs "
                + xs[0]
                + (f"plus_{len(xs)}_more" if len(xs) > 1 else "")
            ).replace(" ", "_")
            + ".png"
        )
    )
    plt.show()


def plot_all_relevant_properties(properties):
    # plot_properties(properties, y='second-dominant TM singular value', x='second-dominant TM eigenvalue magnitude')
    # plot_properties(properties, ys='entanglement entropy', xs='time')
    # plot_properties(properties, ys='entanglement entropy', xs='average bond dimension')
    # plot_properties(properties, ys='energy difference', xs='time')
    # plot_properties(properties, ys='truncation error', xs='time')
    # plot_properties(properties, ys='correlation length', xs='truncation error')
    # plot_properties(properties, ys='correlation length', xs='entanglement entropy')
    # plot_properties(properties, ys='correlation length', xs='average bond dimension')
    # plot_properties(properties, ys='correlation length', xs='time')
    # plot_properties(properties, ys='correlation length', xs='entanglement entropy')
    # plot_properties(properties, ys='correlation length', xs='average bond dimension')
    plot_properties(
        properties, ys="second-dominant TM eigenvalue magnitude", xs="average bond dimension"
    )
    # # plot_properties(properties, ys='validation correlation length', xs='time')

    # # plot_properties(properties, ys='correlation length from svals', xs='correlation length')
    # plot_properties(properties, ys='tenpy correlation length', xs='correlation length')
    # plot_properties(properties, ys='validation correlation length', xs='correlation length')
    # # plot_properties(properties, ys='second-dominant TM singular value', xs='validation second-dominant TM singular value')
    # plot_properties(properties, ys='second-dominant TM eigenvalue magnitude', xs='validation second-dominant TM eigenvalue magnitude')
    # # plot_properties(properties, ys='second-dominant TM singular value', xs='second-dominant TM eigenvalue magnitude')

    # # plot_properties(properties, ys='second-dominant TM singular value', xs='average bond dimension')
    # # plot_properties(properties, ys='second-dominant TM singular value', xs='time')
    # # plot_properties(properties, ys='second-dominant TM singular value', xs='entanglement entropy')
    # # plot_properties(properties, ys='second-dominant TM singular value', xs='truncation error')
    # plot_properties(properties, ys='second-dominant TM eigenvalue imag part', xs='second-dominant TM eigenvalue real part')
    # plot_properties(properties, ys='second-dominant TM eigenvalue magnitude', xs='average bond dimension')
    # plot_properties(properties, ys='second-dominant TM eigenvalue magnitude', xs='time')
    # plot_properties(properties, ys='second-dominant TM eigenvalue magnitude', xs='entanglement entropy')
    # plot_properties(properties, ys='second-dominant TM eigenvalue magnitude', xs='truncation error')
    # plot_properties(properties, ys='second-dominant TM eigenvalue real part', xs='average bond dimension')
    # plot_properties(properties, ys='second-dominant TM eigenvalue real part', xs='time')
    # plot_properties(properties, ys='second-dominant TM eigenvalue real part', xs='entanglement entropy')
    # plot_properties(properties, ys='second-dominant TM eigenvalue real part', xs='truncation error')


if __name__ == "__main__":
    pickles = glob.glob(str(CURRENT_PATH / "pickles/*/*.pickle"))

    properties_list = []
    i = 0
    for file in pickles:
        with open(file, "rb") as f:
            props_here = pickle.load(f)
            props_here["trial number"] += 5.0 * i
            props_here["file"] = file
            properties_list.append(props_here)
            i += 1
    properties = pd.concat(properties_list, ignore_index=True)

    props = properties
    # props = properties[properties['model'] != 'noah']
    # props = props[props['model'] != 'clifford']
    # props = props[(props['trial number'] != 0.0) | (props['model'] != 'aklt')]
    # bad_Js = props[props['time'] > 20]['J']
    # bad_Js = props[props['correlation length'] > 1e10]['J']
    # props = props[~props['J'].isin(bad_Js)]
    props = props[props["time"] < 5]
    # props = props[props['correlation length'] > np.log(props['average bond dimension']) + 2]
    props = props[props["correlation length"] > 5]
    plot_all_relevant_properties(props)

# %%


MODELS = {
    # 'noah': lambda x: 'noah', # Noah's model of random permutations and Hadamards
    "clifford": lambda x: "clifford",  # Noah's model of random permutations and Hadamards
    "tf_ising": tenpy.models.tf_ising.TFIChain,  # Prototypical example of a quantum model: the transverse field Ising model.
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


def run_random(model_name, config_dict=None, **kwargs):
    rs = np.random.randn(100) * 8
    hadamard_rate = np.random.choice([0.0, 0.25, 1.0])
    cnot_rate = np.random.choice([0.0, 0.5, 1.0])
    s_rate = np.random.choice([0.0, 0.25])
    if config_dict is None:
        config_dict = {}
    config_dict.update(
        {
            "model": model_name,
            "bc_MPS": BC_MPS,
            "L": N_SITES,
            "coarsegrain_size": COARSEGRAIN_SIZE,
            "coarsegrain_from": COARSEGRAIN_FROM,
            "j": rs[0],
            "g": rs[1],
            "D": rs[2],
            "E": rs[3],
            "hx": rs[4],
            "hy": rs[5],
            "hz": rs[6],
            "Jx": rs[7],
            "Jy": rs[8],
            "Jz": rs[9],
            "muJ": rs[10],
            "Jxp": rs[11],
            "Jyp": rs[12],
            "Jzp": rs[13],
            "mu": rs[14],
            "V": rs[15],
            "U": rs[16],
            "t": rs[17],
            "t1": rs[18],
            "t2": rs[19],
            "Jp": rs[20],
            "Jv": rs[21],
            "Jxx": rs[22],
            "J": rs[0],
            "hadamard_rate": hadamard_rate,
            "cnot_rate": cnot_rate,
            "s_rate": s_rate,
            "conserve": None,
            "sort_charge": False,
        }
    )
    config = tenpy.tools.params.Config(config_dict, "model_config")
    print(config)
    model = MODELS[config["model"]](config)
    # Set the initial wavefunction
    if isinstance(model, str):
        sites = [tenpy.networks.site.SpinHalfSite("None")] * N_SITES
    else:
        sites = model.lat.mps_sites()
    if model_name == "noah":  # or model_name == 'clifford':
        psi = tenpy.networks.mps.MPS.from_product_state(
            sites,
            [[0.0, 1.0]] * ((N_SITES - 1) // 2)
            + [[1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]]
            + [[0.0, 1.0]] * ((N_SITES - 1) - (N_SITES - 1) // 2),
            bc=BC_MPS,
        )
    else:
        psi = tenpy.networks.mps.MPS.from_desired_bond_dimension(sites, 1, bc=BC_MPS)
    initial_mps_tensors = []
    initial_mps_svals = []
    for n in range(N_SITES):
        initial_mps_tensors.append(psi.get_B(n).to_ndarray())
        initial_mps_svals.append(psi.get_SL(n))
    config_dict["initial_mps_tensors"] = initial_mps_tensors
    config_dict["initial_mps_svals"] = initial_mps_svals

    # Run TEBD time evolution
    psi, model, properties_over_time = run_TEBD(
        psi, model=model, config_dict=config_dict, hadamard_rate=hadamard_rate, **kwargs
    )
    return psi, model, properties_over_time


# %%


if __name__ == "__main__":
    properties = pd.DataFrame(
        columns=[
            "model",
            "time",
            "correlation length",
            "entanglement entropy",
            "average bond dimension",
        ]
    )

    random_samples = 10
    for s in range(random_samples):
        for model_name in MODELS:  # ['aklt']:
            p = properties[properties["time"] == 0]
            config_dict = {}
            config_dict["total trial number"] = len(p)
            config_dict["trial number"] = len(p[p["model"] == model_name])

            psi, model, properties_over_time = run_random(model_name, config_dict=config_dict)
            properties = pd.concat([properties, properties_over_time], ignore_index=True)

            # Save results
            with open(PICKLE_DIR / "properties_df.pickle", "wb") as f:
                pickle.dump(properties, f)
            with open(PICKLE_DIR / "properties.csv", "w") as f:
                f.write(properties.to_csv())

            # Plot results
            plot_all_relevant_properties(properties)


# %%

# def plot_properties(properties, y='correlation length', x='entanglement entropy'):
#     plt.rcParams["figure.figsize"] = (8,8)
#     if y == 'correlation length':
#         y_data = abs(properties[y])
#         plt.ylim((0.0,40.0))
#     else:
#         y_data = y
#     sns.lineplot(properties, y=y_data, x=x, hue='model', style='trial number', legend=False, alpha=0.35)
#     sns.scatterplot(properties, y=y_data, x=x, hue='model', style='trial number')
#     plt.savefig(PLOTS_DIR / ((y + ' vs ' + x).replace(' ','_') + '.png'))
#     plt.show()

# def plot_all_relevant_properties(properties):
#     # plot_properties(properties, y='energy', x='time')
#     # plot_properties(properties, y='energy difference', x='time')
#     # plot_properties(properties, y='truncation error', x='time')
#     plot_properties(properties, y='correlation length', x='truncation error')
#     plot_properties(properties, y='correlation length', x='entanglement entropy')
#     plot_properties(properties, y='correlation length', x='average bond dimension')
#     plot_properties(properties, y='correlation length', x='time')
#     # plot_properties(properties, y='entanglement entropy', x='time')
#     # plot_properties(properties, y='second-dominant TM eigenvalue imag part', x='second-dominant TM eigenvalue real part')
#     # plot_properties(properties, y='second-dominant TM eigenvalue magnitude', x='average bond dimension')
#     # plot_properties(properties, y='second-dominant TM eigenvalue magnitude', x='time')
#     # plot_properties(properties, y='second-dominant TM eigenvalue magnitude', x='entanglement entropy')
#     # plot_properties(properties, y='second-dominant TM eigenvalue magnitude', x='truncation error')
#     # plot_properties(properties, y='second-dominant TM eigenvalue real part', x='average bond dimension')
#     # plot_properties(properties, y='second-dominant TM eigenvalue real part', x='time')
#     # plot_properties(properties, y='second-dominant TM eigenvalue real part', x='entanglement entropy')
#     # plot_properties(properties, y='second-dominant TM eigenvalue real part', x='truncation error')

# plot_all_relevant_properties(properties)
# %%


# Save the tensors when the correlation length > 5 (when |2nd TM eval| > 0.65), when the bond dimension first gets above 25, 50, 100, 150.
