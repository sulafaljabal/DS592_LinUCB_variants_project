"""
experiments/sensitivity.py — Hyperparameter sensitivity analysis
=================================================================
Sweep over γ (for D-LinUCB) and τ (for SW-LinUCB) to understand
how sensitive regret is to these hyperparameters under different
drift types. This directly addresses the proposal's goal of
understanding "sensitivity to hyperparameters of the model."

Run from the project root:
    python -m experiments.sensitivity
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms import DLinUCB, SWLinUCB
from drift_functions import paper_abrupt_drift, slow_sinusoidal, fast_linear_drift
from environments import NonStationaryLinearBandit, run_experiment
from utils import plot_regret_vs_parameter
from config import *


def sweep_gamma(drift_fn, drift_label, actions, gammas):
    """Sweep discount factor γ for D-LinUCB."""
    print(f"\n  Sweeping γ for D-LinUCB on {drift_label}...")
    results = []
    for gamma in gammas:
        algo = DLinUCB(d=D, gamma=gamma, lambda_reg=LAMBDA_REG, delta=DELTA)
        res = run_experiment(
            algorithm=algo,
            environment=NonStationaryLinearBandit(
                d=D, drift_fn=drift_fn, n_actions=N_ACTIONS,
                sigma_noise=NOISE_STD, fixed_actions=actions,
            ),
            T=T,
            n_simulations=N_SIMULATIONS // 2,  # fewer sims for speed
            seed=SEED,
        )
        final_mean = res['mean_regret'][-1]
        final_std = res['std_regret'][-1]
        results.append((final_mean, final_std))
        print(f"    γ={gamma:.4f}  →  regret = {final_mean:.1f} ± {final_std:.1f}")
    return results


def sweep_tau(drift_fn, drift_label, actions, taus):
    """Sweep window size τ for SW-LinUCB."""
    print(f"\n  Sweeping τ for SW-LinUCB on {drift_label}...")
    results = []
    for tau in taus:
        algo = SWLinUCB(d=D, tau=tau, lambda_reg=LAMBDA_REG, delta=DELTA)
        res = run_experiment(
            algorithm=algo,
            environment=NonStationaryLinearBandit(
                d=D, drift_fn=drift_fn, n_actions=N_ACTIONS,
                sigma_noise=NOISE_STD, fixed_actions=actions,
            ),
            T=T,
            n_simulations=N_SIMULATIONS // 2,
            seed=SEED,
        )
        final_mean = res['mean_regret'][-1]
        final_std = res['std_regret'][-1]
        results.append((final_mean, final_std))
        print(f"    τ={tau}  →  regret = {final_mean:.1f} ± {final_std:.1f}")
    return results


def main():
    print("=" * 60)
    print("Hyperparameter Sensitivity Analysis")
    print("=" * 60)

    # Fixed action set
    rng_actions = np.random.RandomState(42)
    actions = rng_actions.randn(N_ACTIONS, D)
    actions = actions / np.linalg.norm(actions, axis=1, keepdims=True)

    # Parameter grids
    gammas = [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
    taus = [50, 100, 200, 400, 800, 1500, 3000]

    # Drift functions to test
    drifts = {
        'Abrupt':          paper_abrupt_drift(d=D),
        'Slow Sinusoidal': slow_sinusoidal(d=D, T=T),
        'Fast Linear':     fast_linear_drift(d=D, T=T),
    }

    for drift_label, drift_fn in drifts.items():
        print(f"\n{'=' * 50}")
        print(f"Drift: {drift_label} (B_T = {drift_fn.total_variation(T):.4f})")
        print(f"{'=' * 50}")

        # ─── Sweep γ ───
        gamma_results = sweep_gamma(drift_fn, drift_label, actions, gammas)
        safe_label = drift_label.lower().replace(' ', '_')

        plot_regret_vs_parameter(
            param_values=gammas,
            regret_values={'D-LinUCB': gamma_results},
            param_name='Discount factor γ',
            algo_names=['D-LinUCB'],
            title=f"D-LinUCB: Regret vs γ — {drift_label} Drift",
            save_path=os.path.join(PLOT_DIR, f"sensitivity_gamma_{safe_label}.png"),
        )

        # ─── Sweep τ ───
        tau_results = sweep_tau(drift_fn, drift_label, actions, taus)

        plot_regret_vs_parameter(
            param_values=taus,
            regret_values={'SW-LinUCB': tau_results},
            param_name='Window size τ',
            algo_names=['SW-LinUCB'],
            title=f"SW-LinUCB: Regret vs τ — {drift_label} Drift",
            save_path=os.path.join(PLOT_DIR, f"sensitivity_tau_{safe_label}.png"),
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
