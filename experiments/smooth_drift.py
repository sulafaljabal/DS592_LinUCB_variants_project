"""
experiments/smooth_drift.py — Experiments with smooth drift functions
======================================================================
This is the NOVEL part of the project: testing D-LinUCB and SW-LinUCB
on smooth/structured drift instead of the sharp breakpoints from the paper.

Drift functions tested:
  - Slow linear drift
  - Fast linear drift
  - Slow sinusoidal drift
  - Fast sinusoidal drift

Run from the project root:
    python -m experiments.smooth_drift
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms import LinUCB, DLinUCB, SWLinUCB
from drift_functions import (
    slow_linear_drift, fast_linear_drift,
    slow_sinusoidal, fast_sinusoidal,
)
from environments import NonStationaryLinearBandit, run_experiment
from utils import plot_cumulative_regret, plot_theta_trajectory
from config import *


def run_drift_experiment(drift_fn, label, actions):
    """Run all algorithms on a single drift function."""
    B_T = drift_fn.total_variation(T)
    gamma_opt = compute_optimal_gamma(B_T, D, T)
    tau_opt = compute_optimal_tau(B_T, D, T)

    print(f"\n{'─' * 50}")
    print(f"Drift: {label}")
    print(f"Total variation B_T = {B_T:.4f}")
    print(f"Optimal γ = {gamma_opt:.6f}, Optimal τ = {tau_opt}")
    print(f"{'─' * 50}")

    env = NonStationaryLinearBandit(
        d=D, drift_fn=drift_fn, n_actions=N_ACTIONS,
        sigma_noise=NOISE_STD, fixed_actions=actions,
    )

    algos = {
        'LinUCB': LinUCB(d=D, lambda_reg=LAMBDA_REG, delta=DELTA),
        'D-LinUCB': DLinUCB(d=D, gamma=gamma_opt, lambda_reg=LAMBDA_REG, delta=DELTA),
        'SW-LinUCB': SWLinUCB(d=D, tau=tau_opt, lambda_reg=LAMBDA_REG, delta=DELTA),
    }

    results = {}
    for name, algo in algos.items():
        print(f"  Running {name}...")
        results[name] = run_experiment(
            algorithm=algo, environment=env,
            T=T, n_simulations=N_SIMULATIONS, seed=SEED,
        )
        print(f"    Final mean regret: {results[name]['mean_regret'][-1]:.1f}")

    return results


def main():
    print("=" * 60)
    print("Smooth Drift Experiments")
    print("Testing D-LinUCB on linear and sinusoidal drift")
    print("=" * 60)

    # Fixed action set for fair comparison across drifts
    rng_actions = np.random.RandomState(42)
    actions = rng_actions.randn(N_ACTIONS, D)
    actions = actions / np.linalg.norm(actions, axis=1, keepdims=True)

    # ─── Define all drift functions ───
    drifts = {
        'Slow Linear':      slow_linear_drift(d=D, T=T),
        'Fast Linear':      fast_linear_drift(d=D, T=T),
        'Slow Sinusoidal':  slow_sinusoidal(d=D, T=T),
        'Fast Sinusoidal':  fast_sinusoidal(d=D, T=T),
    }

    # ─── Run and plot each ───
    all_results = {}
    for label, drift_fn in drifts.items():
        results = run_drift_experiment(drift_fn, label, actions)
        all_results[label] = results

        safe_label = label.lower().replace(' ', '_')

        plot_cumulative_regret(
            results, T=T,
            title=f"Cumulative Regret — {label} Drift",
            save_path=os.path.join(PLOT_DIR, f"smooth_{safe_label}_regret.png"),
        )

        plot_theta_trajectory(
            results, drift_fn=drift_fn, T=T, d=D,
            title=f"Parameter Tracking — {label} Drift",
            save_path=os.path.join(PLOT_DIR, f"smooth_{safe_label}_theta.png"),
        )

    # ─── Summary comparison: final regret across drift types ───
    print("\n" + "=" * 60)
    print("SUMMARY: Final mean regret across drift types")
    print("=" * 60)
    header = f"{'Drift Type':<20} {'LinUCB':>10} {'D-LinUCB':>10} {'SW-LinUCB':>12} {'B_T':>10}"
    print(header)
    print("-" * len(header))

    for label, drift_fn in drifts.items():
        results = all_results[label]
        bv = drift_fn.total_variation(T)
        row = f"{label:<20}"
        for algo_name in ['LinUCB', 'D-LinUCB', 'SW-LinUCB']:
            final = results[algo_name]['mean_regret'][-1]
            row += f" {final:>10.1f}"
        row += f" {bv:>10.4f}"
        print(row)

    print("\nDone!")


if __name__ == "__main__":
    main()
