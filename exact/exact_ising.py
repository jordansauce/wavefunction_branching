# %%
###############################################################################################
# ANALYTIC EXPECTATION VALUES IN THE ISING MODEL
###############################################################################################
import glob
from pathlib import Path
from typing import Literal

import fire
import matplotlib.pyplot as plt
import numpy as np
from tqdm.autonotebook import tqdm

RESULTS_PATH = Path("exact/results/")


class IsingAnalytic:
    """Adapted from https://github.com/mgbukov/MBQD/blob/main/notebooks/Integrable%20Quench%20Dynamics.ipynb"""

    def __init__(self, J=1.0, g=2.0, g_0=np.inf, L=50, flag=False):
        assert L % 2 == 0, "L must be even"
        self.J = J  # Ising model overall strength
        self.g = g  # Ising model transverse field strength
        self.g_0 = g_0  # Ising model transverse field strength at t=0 (for the quench)
        self.L = L  # Length of chain (periodic boundary conditions)
        self.flag = flag

    # Functions for exact solution
    def epsilon_h(self, k):
        J = self.J
        g = self.g
        return 2 * np.sqrt((J**2 + g**2) - 2.0 * J * g * np.cos(k))

    def theta(self, k):
        J = self.J
        g = self.g
        return np.arctan((-J * np.sin(k)) / (g - J * np.cos(k)))

    def Delta(self, k):
        J = self.J
        g = self.g
        g_0 = self.g_0
        return np.arctan((-J * np.sin(k)) / (g - J * np.cos(k))) - np.arctan(
            (-J * np.sin(k)) / (g_0 - J * np.cos(k))
        )

    def Ak(self, k, t):
        res = np.cos(self.theta(k) / 2.0) * np.cos(self.Delta(k) / 2.0) * np.exp(
            -1j * self.epsilon_h(k) * t
        ) + np.sin(self.theta(k) / 2.0) * np.sin(self.Delta(k) / 2.0) * np.exp(
            +1j * self.epsilon_h(k) * t
        )
        return res

    def Bk(self, k, t):
        res = 1j * np.cos(self.theta(k) / 2.0) * np.sin(self.Delta(k) / 2.0) * np.exp(
            +1j * self.epsilon_h(k) * t
        ) - 1j * np.sin(self.theta(k) / 2.0) * np.cos(self.Delta(k) / 2.0) * np.exp(
            -1j * self.epsilon_h(k) * t
        )
        return res

    def sigma_x(self, t):
        L = self.L
        k_range = (0.5 + np.arange(L)) * 2.0 * np.pi / L
        return -(2.0 * np.sum([np.abs(self.Bk(k_, t)) ** 2 for k_ in k_range]).real / L - 1.0)

    def sigma_x_over_time(self, t_start=0.0, t_end=10.0, timestep=0.025):
        ts = np.arange(t_start, t_end, timestep)
        Szs = np.zeros_like(ts)
        for i, t in tqdm(enumerate(ts), total=len(ts)):
            Szs[i] = self.sigma_x(t)
        return ts, Szs

    def sigma_x_sigma_x(self, t, d):
        # In the notation of the lecture notes, this is the
        # expression for
        # (1/L) < \sum_j \sigma^z_j sigma^z_{j+d} >
        # where j is a discrete spatial coordinate and d
        # is the distance.
        # I did not insert a minus sign to try and match with
        # sigma_x above because I figure you'd need one for
        # each sigma_x and they'd cancel
        L = self.L
        k_range = (0.5 + np.arange(L)) * 2.0 * np.pi / L
        Bs = {}
        As = {}
        # test = np.zeros_like(k_range)
        for k_ in k_range:
            Bs[k_] = self.Bk(k_, t)
            Bs[-k_] = self.Bk(-k_, t)
            As[k_] = self.Ak(k_, t)
            As[-k_] = self.Ak(-k_, t)
            # print(f'k={k_}, test quantity = {np.conj(Bs[k_])*As[k_] - np.conj(Bs[-k_])*As[-k_]}')

        # The following should be real, but it's not manifestly so.
        # Could do error checking on this

        val1 = 0.0j
        val2 = 0.0j
        val3 = 0.0j
        for k_ in k_range:
            for p_ in k_range:
                if p_ != k_:
                    val1 += (
                        np.exp(1j * (k_ - p_) * d)
                        * Bs[p_]
                        * np.conj(As[k_])
                        * ((-1) * np.conj(Bs[-k_]) * As[-p_] + np.conj(Bs[p_]) * As[k_])
                    )
                    val1 += (
                        (1 + 0 * np.exp(1j * (p_ - k_) * d))
                        * (np.abs(Bs[k_]) ** 2)
                        * (np.abs(Bs[p_]) ** 2)
                    )

            val2 += np.abs(Bs[k_]) ** 2

            val3 += np.abs(Bs[k_]) ** 4

        return 1.0 + ((2.0 / L) ** 2) * val1 - (4.0 / L) * val2 + (4.0 / L**2) * val3

    def sigma_x_sigma_x_over_time(self, d, t_start=0.0, t_end=10.0, timestep=0.025):
        ts = np.arange(t_start, t_end, timestep)
        SzSzs = np.zeros_like(ts, dtype=complex)
        for i, t in tqdm(enumerate(ts), total=len(ts)):
            SzSzs[i] = self.sigma_x_sigma_x(t, d)
        return ts, SzSzs.real


def save_sigma_x_sigma_x(L, d, t_end=20.0, timestep=0.2, save=True, plot=True):
    analytic = IsingAnalytic(L=L)
    ts, SzSzs = analytic.sigma_x_sigma_x_over_time(d, t_start=0, t_end=t_end, timestep=timestep)
    results = np.stack([ts, SzSzs], axis=1)

    # Save results
    if save:
        with open(RESULTS_PATH / f"sigma_x_sigma_x_distance_{d}_L_{L}.npy", "wb") as f:
            np.save(f, results)
        with open(RESULTS_PATH / f"sigma_x_sigma_x_distance_{d}_L_{L}.txt", "w") as f:
            np.savetxt(f, results)

    print(f"saved to {RESULTS_PATH}/sigma_x_sigma_x_distance_{d}_L_{L}.npy.")


def save_sigma_x(L, t_end=20.0, timestep=0.2, save=True, plot=True):
    analytic = IsingAnalytic(L=L)
    ts, SzSzs = analytic.sigma_x_over_time(t_start=0, t_end=t_end, timestep=timestep)
    results = np.stack([ts, SzSzs], axis=1)

    # Save results
    if save:
        with open(RESULTS_PATH / f"sigma_x_L_{L}.npy", "wb") as f:
            np.save(f, results)
        with open(RESULTS_PATH / f"sigma_x_L_{L}.txt", "w") as f:
            np.savetxt(f, results)

    print(f"saved to {RESULTS_PATH}/sigma_x_L_{L}.npy.")


def plot_operator(operator, d):
    plt.rcParams["figure.figsize"] = [23, 13]
    dist_str = "" if d == 0 else f"_distance_{d}"
    plt.figure(dpi=120)
    all_results_files = glob.glob(str(RESULTS_PATH / f"{operator}{dist_str}_L*.npy"))
    for results_file in all_results_files:
        result = np.load(results_file)
        L_ = int(results_file.split("L_")[1].split(".")[0])
        plt.plot(
            result[:, 0],
            result[:, 1],
            alpha=0.45,
            marker="+",
            linestyle="--",
            label=f"d = {d}, L={L_}",
        )
    plt.xlabel("t")
    plt.ylabel(ylabel=f"<{operator}>{dist_str.replace('_', ' ')}")
    plt.legend()
    plt.savefig(RESULTS_PATH / f"{operator}{dist_str}.png")
    plt.savefig(RESULTS_PATH / f"{operator}{dist_str}.pdf")
    print(f"saved to {RESULTS_PATH}/{operator}{dist_str}.png")
    plt.show()


def main(
    operator: Literal["sigma_x", "sigma_x_sigma_x"] = "sigma_x_sigma_x",
    L: int = 100,
    d: int = 0,
    t_end=20.0,
    timestep=0.025,
    save=True,
    plot=True,
):
    print("Calculating analytic expectation values:")
    if operator == "sigma_x":
        save_sigma_x(L, t_end=t_end, timestep=timestep, save=save, plot=plot)
    if operator == "sigma_x_sigma_x":
        save_sigma_x_sigma_x(L, d, t_end=t_end, timestep=timestep, save=save, plot=plot)
    print(f"Done.\n\nPlotting {operator} (distance {d}): ")
    plot_operator(operator, 0 if operator == "sigma_x" else d)
    print("Finished.")


def print_args_kwargs_then_run_main(*args, **kwargs):
    if len(args) > 0:
        print("args: ")
        for i in range(len(args)):
            print(f"    arg {i}:  {args[i]}  |  type = {type(args[i]).__name__}")
    if len(kwargs) > 0:
        print("kwargs: ")
        for k, v in kwargs.items():
            print(f"    {k}:  {v}  |  type = {type(v).__name__}")
    main(*args, **kwargs)


if __name__ == "__main__":
    RESULTS_PATH.mkdir(exist_ok=True)
    fire.Fire(print_args_kwargs_then_run_main)
    # main('sigma_x_sigma_x', L=30, d=1)
