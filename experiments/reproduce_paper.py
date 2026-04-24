"""
experiments/reproduce_paper.py — Reproduce Russac et al. Figure 1
==================================================================
This script runs LinUCB, D-LinUCB, and SW-LinUCB on the abruptly-changing
environment from the paper (d=2, T=6000, 3 breakpoints, N=100 runs).

Run from the project root:
    python -m experiments.reproduce_paper
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms import LinUCB, DLinUCB, SWLinUCB, DynamicLinUCB
from drift_functions import paper_abrupt_drift
from environments import NonStationaryLinearBandit, run_experiment
from utils import plot_cumulative_regret, plot_theta_trajectory
from config import *


def main():
    print("=" * 60)
    print("Reproducing Russac et al. (2019) Figure 1")
    print("Abruptly-changing environment, d=2, T=6000")
    print("=" * 60)

    # ─── Setup ───
    drift = paper_abrupt_drift(d=D)
    B_T = drift.total_variation(T)
    print(f"\nDrift: {drift}")
    print(f"Total variation B_T = {B_T:.2f}")

    # Compute theoretically optimal γ and τ (Corollary 1 of Russac et al.)
    gamma_opt = compute_optimal_gamma(B_T, D, T)
    tau_opt = compute_optimal_tau(B_T, D, T)
    print(f"Optimal γ = 1 - (B_T/(d*T))^(2/3) = {gamma_opt:.6f}")
    print(f"Optimal τ = (d*T/B_T)^(2/3) = {tau_opt}")

    # Fixed action set: unit vectors on the circle (like the paper)
    rng_actions = np.random.RandomState(42)
    actions = rng_actions.randn(N_ACTIONS, D)
    actions = actions / np.linalg.norm(actions, axis=1, keepdims=True)

    env = NonStationaryLinearBandit(
        d=D, drift_fn=drift, n_actions=N_ACTIONS,
        sigma_noise=NOISE_STD, fixed_actions=actions,
    )

    # ─── Algorithms ───
    algos = {
        'LinUCB': LinUCB(d=D, lambda_reg=LAMBDA_REG, delta=DELTA),
        'D-LinUCB': DLinUCB(d=D, gamma=gamma_opt, lambda_reg=LAMBDA_REG, delta=DELTA),
        'SW-LinUCB': SWLinUCB(d=D, tau=tau_opt, lambda_reg=LAMBDA_REG, delta=DELTA),
        'dLinUCB': DynamicLinUCB(d=D, tau=tau_opt, lambda_reg=LAMBDA_REG, delta=DELTA),
    }

    # ─── Run experiments ───
    results = {}
    for name, algo in algos.items():
        print(f"\nRunning {name}...")
        results[name] = run_experiment(
            algorithm=algo,
            environment=env,
            T=T,
            n_simulations=N_SIMULATIONS,
            seed=SEED,
        )
        final_regret = results[name]['mean_regret'][-1]
        print(f"  {name} final mean regret: {final_regret:.1f}")

    # ─── Plot results ───
    print("\nGenerating plots...")

    plot_cumulative_regret(
        results, T=T,
        title="Cumulative Regret — Abrupt Drift (Reproducing Russac et al.)",
        save_path=os.path.join(PLOT_DIR, "reproduce_abrupt_regret.png"),
    )

    plot_theta_trajectory(
        results, drift_fn=drift, T=T, d=D,
        title="Parameter Tracking — Abrupt Drift",
        save_path=os.path.join(PLOT_DIR, "reproduce_abrupt_theta.png"),
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
