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

# ─── D-LinUCB defaults ───
GAMMA = 0.99         # discount factor for D-LinUCB (tune this!)

# ─── SW-LinUCB defaults ───
TAU = 200            # sliding window size for SW-LinUCB

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
