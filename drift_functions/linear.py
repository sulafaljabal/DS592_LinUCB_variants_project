"""
drift_functions/linear.py — Linear (monotone) drift
=====================================================
θ* moves linearly from a start point to an end point over the horizon.

    θ*_t = θ_start + (t / T) * (θ_end - θ_start)

This represents gradual, monotone change — e.g., a user's preferences
slowly shifting in one direction.
"""

import numpy as np
from drift_functions.base import DriftFunction


class LinearDrift(DriftFunction):
    """θ* drifts linearly from start to end over T timesteps."""

    def __init__(self, d, theta_start, theta_end, T):
        """
        Parameters
        ----------
        d : int
            Dimension.
        theta_start : np.ndarray of shape (d,)
            Starting parameter.
        theta_end : np.ndarray of shape (d,)
            Ending parameter.
        T : int
            Total horizon (used for normalization).
        """
        super().__init__(d, name="LinearDrift")
        self.theta_start = theta_start.copy()
        self.theta_end = theta_end.copy()
        self.T = T
        self.direction = (theta_end - theta_start) / T

    def __call__(self, t):
        return self.theta_start + t * self.direction


# ─── Convenience constructors ───
def slow_linear_drift(d=2, T=6000):
    """Small linear drift: θ* moves from (1,0) to (0.5, 0.5)."""
    return LinearDrift(
        d=d,
        theta_start=np.array([1.0, 0.0]),
        theta_end=np.array([0.5, 0.5]),
        T=T,
    )

def fast_linear_drift(d=2, T=6000):
    """Large linear drift: θ* moves from (1,0) to (-1, 0)."""
    return LinearDrift(
        d=d,
        theta_start=np.array([1.0, 0.0]),
        theta_end=np.array([-1.0, 0.0]),
        T=T,
    )
