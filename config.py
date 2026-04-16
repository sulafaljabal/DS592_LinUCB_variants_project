"""
config.py — Shared experiment parameters
=========================================
Centralizes all hyperparameters so experiments are reproducible
and easy to tune from one place.
"""

import numpy as np

# ─── Random seed for reproducibility ───
SEED = 592

# ─── Environment defaults ───
D = 2               # dimension of action/parameter space
T = 6000            # time horizon (matches Russac et al. Figure 1)
N_ACTIONS = 10      # number of available actions per round
NOISE_STD = 1.0     # sub-Gaussian noise std (σ)

# ─── Monte Carlo ───
N_SIMULATIONS = 100  # number of independent runs to average

# ─── LinUCB defaults ───
LAMBDA_REG = 1.0     # regularization parameter λ
DELTA = 0.01         # confidence parameter δ

# ─── D-LinUCB / SW-LinUCB theoretical tuning ───
# From Corollary 1 of Russac et al. (2019):
#   γ* = 1 - (B_T / (d * T))^{2/3}
#   τ* = (d * T / B_T)^{2/3}
#
# These require knowing B_T (total variation of θ*). In practice you
# compute B_T from your drift function, then call the helpers below.
# The old hardcoded values (GAMMA=0.99, TAU=200) are kept as fallbacks.

def compute_optimal_gamma(B_T, d, T):
    """
    Optimal discount factor from Corollary 1 of Russac et al. (2019).

        γ* = 1 - (B_T / (d * T))^{2/3}

    Parameters
    ----------
    B_T : float
        Total variation budget Σ ‖θ*_{t+1} - θ*_t‖.
    d : int
        Dimension.
    T : int
        Time horizon.

    Returns
    -------
    float
        Optimal γ ∈ (0, 1). Clipped to [0.5, 0.9999] for numerical safety.
    """
    if B_T <= 0:
        return 0.9999  # essentially stationary → minimal discounting
    gamma = 1.0 - (B_T / (d * T)) ** (2.0 / 3.0)
    return float(np.clip(gamma, 0.5, 0.9999))


def compute_optimal_tau(B_T, d, T):
    """
    Optimal sliding window size from Cheung et al. (2019) / Russac et al.

        τ* = (d * T / B_T)^{2/3}

    Parameters
    ----------
    B_T : float
        Total variation budget.
    d : int
        Dimension.
    T : int
        Time horizon.

    Returns
    -------
    int
        Optimal τ ≥ 1. Clipped to [1, T].
    """
    if B_T <= 0:
        return T  # stationary → use all data
    tau = (d * T / B_T) ** (2.0 / 3.0)
    return int(np.clip(tau, 1, T))


# Legacy fallback values (used only if B_T is unknown)
GAMMA_DEFAULT = 0.99
TAU_DEFAULT = 200

# ─── Drift function defaults ───
# Abrupt drift: breakpoints and theta values (from the paper)
ABRUPT_BREAKPOINTS = [0, 1500, 3000, 4500]  # timesteps where θ* changes
ABRUPT_THETAS = [                             # θ* values at each segment
    np.array([1.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([-1.0, 0.0]),
    np.array([0.0, -1.0]),
]

# Linear drift
LINEAR_DRIFT_SPEED = 0.001   # units of θ* change per timestep

# Sinusoidal drift
SINUSOIDAL_AMPLITUDE = 0.5   # amplitude of oscillation
SINUSOIDAL_PERIOD = 2000     # period in timesteps

# ─── Plot settings ───
PLOT_DIR = "plots/"
COLORS = {
    'LinUCB':     'steelblue',
    'D-LinUCB':   'darkorange',
    'SW-LinUCB':  'forestgreen',
    'dLinUCB':    'firebrick',
}
