"""
drift_functions/abrupt.py — Abrupt (breakpoint) drift
=======================================================
θ* jumps between fixed values at specified timesteps.
This is the "abruptly-changing environment" from Russac et al. Figure 1.
"""

import numpy as np
from drift_functions.base import DriftFunction


class AbruptDrift(DriftFunction):
    """θ* changes abruptly at specified breakpoints."""

    def __init__(self, d, breakpoints, thetas):
        """
        Parameters
        ----------
        d : int
            Dimension.
        breakpoints : list of int
            Timesteps where θ* changes. Must be sorted.
            E.g., [0, 1500, 3000, 4500] means θ* = thetas[0] for t in [1, 1500),
            θ* = thetas[1] for t in [1500, 3000), etc.
        thetas : list of np.ndarray
            Parameter values for each segment. len(thetas) == len(breakpoints).
        """
        assert len(breakpoints) == len(thetas), "Need one θ per breakpoint"
        assert all(t.shape == (d,) for t in thetas), f"All θ must have shape ({d},)"
        super().__init__(d, name="AbruptDrift")
        self.breakpoints = breakpoints
        self.thetas = thetas

    def __call__(self, t):
        """Return θ* at timestep t."""
        # Find which segment t falls into
        idx = 0
        for i, bp in enumerate(self.breakpoints):
            if t >= bp:
                idx = i
        return self.thetas[idx].copy()


# ─── Convenience: default abrupt drift from the paper ───
def paper_abrupt_drift(d=2):
    """
    Recreate the abruptly-changing environment from Russac et al. Fig 1.
    d=2, T=6000, 3 breakpoints, θ* jumps around the unit circle.
    """
    breakpoints = [0, 1500, 3000, 4500]
    thetas = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, -1.0]),
    ]
    return AbruptDrift(d=d, breakpoints=breakpoints, thetas=thetas)
