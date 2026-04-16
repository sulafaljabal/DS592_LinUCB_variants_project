"""
drift_functions/sinusoidal.py — Sinusoidal (periodic) drift
=============================================================
θ* oscillates periodically, modeling cyclical preference changes
(e.g., seasonal interest shifts).

    θ*_t = θ_center + amplitude * [cos(2πt/period), sin(2πt/period), ...]

This is one of the key novel drift types from your proposal.
"""

import numpy as np
from drift_functions.base import DriftFunction


class SinusoidalDrift(DriftFunction):
    """θ* oscillates sinusoidally around a center point."""

    def __init__(self, d, center, amplitude, period):
        """
        Parameters
        ----------
        d : int
            Dimension.
        center : np.ndarray of shape (d,)
            Center of oscillation.
        amplitude : float
            Amplitude of oscillation (applied to unit-norm direction).
        period : int or float
            Period in timesteps.
        """
        super().__init__(d, name=f"SinusoidalDrift(A={amplitude}, P={period})")
        self.center = center.copy()
        self.amplitude = amplitude
        self.period = period

    def __call__(self, t):
        """θ* traces a circle (in 2D) or oscillates along first 2 coords."""
        theta = self.center.copy()
        phase = 2 * np.pi * t / self.period
        # Oscillate in the first two dimensions
        if self.d >= 1:
            theta[0] += self.amplitude * np.cos(phase)
        if self.d >= 2:
            theta[1] += self.amplitude * np.sin(phase)
        return theta


# ─── Convenience constructors ───
def slow_sinusoidal(d=2, T=6000):
    """Slow oscillation: one full period over the horizon."""
    return SinusoidalDrift(
        d=d,
        center=np.zeros(d),
        amplitude=0.5,
        period=T,
    )

def fast_sinusoidal(d=2, T=6000):
    """Fast oscillation: multiple periods over the horizon."""
    return SinusoidalDrift(
        d=d,
        center=np.zeros(d),
        amplitude=0.5,
        period=T // 6,  # 6 full cycles
    )
