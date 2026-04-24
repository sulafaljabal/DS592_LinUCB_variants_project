"""
utils/plotting.py — Shared plotting utilities
===============================================
Consistent plot style across all experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_cumulative_regret(results_dict, T, title="Cumulative Regret",
                           save_path=None, show_std=True):
    """
    Plot cumulative regret curves for multiple algorithms.

    Parameters
    ----------
    results_dict : dict
        Keys are algorithm names, values are dicts from run_experiment().
    T : int
        Time horizon.
    title : str
        Plot title.
    save_path : str or None
        If provided, save the figure to this path.
    show_std : bool
        If True, show ±1 std shaded region.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    timesteps = np.arange(1, T + 1)

    colors = ['steelblue', 'darkorange', 'forestgreen', 'firebrick',
              'mediumpurple', 'brown']

    for i, (name, res) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        mean = res['mean_regret']
        ax.plot(timesteps, mean, label=name, color=color, linewidth=1.5)

        if show_std:
            std = res['std_regret']
            ax.fill_between(timesteps, mean - std, mean + std,
                            color=color, alpha=0.15)

    ax.set_xlabel('Time step t', fontsize=12)
    ax.set_ylabel('Cumulative Regret', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show(block=False)


def plot_theta_trajectory(results_dict, drift_fn, T, d=2,
                          title="Parameter Estimates vs True θ*",
                          save_path=None):
    """
    Plot how θ̂_t tracks the true θ*_t over time.

    Parameters
    ----------
    results_dict : dict
        Keys are algorithm names, values are dicts from run_experiment().
    drift_fn : DriftFunction
        The true drift function (to plot ground truth).
    T : int
        Time horizon.
    d : int
        Dimension (only plots first 2 components).
    """
    timesteps = np.arange(1, T + 1)

    # True θ* trajectory
    true_theta = np.array([drift_fn(t) for t in timesteps])

    fig, axes = plt.subplots(min(d, 2), 1, figsize=(10, 4 * min(d, 2)),
                             sharex=True)
    if min(d, 2) == 1:
        axes = [axes]

    colors = ['steelblue', 'darkorange', 'forestgreen', 'firebrick']

    for dim_idx in range(min(d, 2)):
        ax = axes[dim_idx]
        # True value
        ax.plot(timesteps, true_theta[:, dim_idx], 'k--', linewidth=2,
                label='True θ*', alpha=0.7)

        for i, (name, res) in enumerate(results_dict.items()):
            # Mean estimate across simulations
            mean_est = np.mean(res['theta_estimates'][:, :, dim_idx], axis=0)
            ax.plot(timesteps, mean_est, color=colors[i % len(colors)],
                    linewidth=1, label=name, alpha=0.8)

        ax.set_ylabel(f'θ[{dim_idx}]', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time step t', fontsize=12)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_regret_vs_parameter(param_values, regret_values, param_name,
                              algo_names, title="Regret vs Parameter",
                              save_path=None):
    """
    Plot final regret as a function of a swept parameter.
    Useful for hyperparameter sensitivity analysis.

    Parameters
    ----------
    param_values : list or np.ndarray
        The parameter values swept.
    regret_values : dict
        Keys are algorithm names, values are lists of (mean, std) tuples.
    param_name : str
        Name of the parameter (for x-axis label).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['steelblue', 'darkorange', 'forestgreen', 'firebrick']

    for i, name in enumerate(algo_names):
        means = [r[0] for r in regret_values[name]]
        stds = [r[1] for r in regret_values[name]]
        color = colors[i % len(colors)]
        ax.errorbar(param_values, means, yerr=stds, label=name,
                     color=color, marker='o', capsize=3, linewidth=1.5)

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Final Cumulative Regret', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
