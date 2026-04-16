"""
drift_functions/piecewise.py — Piecewise linear drift
=======================================================
θ* follows a piecewise linear path through a sequence of waypoints.
This is a middle ground between abrupt (instant jumps) and smooth
(continuous functions): θ* changes linearly between waypoints,
creating "ramps" instead of jumps.
"""

import numpy as np
from drift_functions.base import DriftFunction


class PiecewiseLinearDrift(DriftFunction):
    """θ* interpolates linearly between a sequence of waypoints."""

    def __init__(self, d, waypoints, times):
        """
        Parameters
        ----------
        d : int
            Dimension.
        waypoints : list of np.ndarray
            Sequence of θ* values to interpolate between.
        times : list of int
            Timesteps at which each waypoint is reached.
            Must be sorted. len(times) == len(waypoints).
        """
        assert len(waypoints) == len(times)
        super().__init__(d, name="PiecewiseLinearDrift")
        self.waypoints = [w.copy() for w in waypoints]
        self.times = times

    def __call__(self, t):
        # Before first waypoint
        if t <= self.times[0]:
            return self.waypoints[0].copy()
        # After last waypoint
        if t >= self.times[-1]:
            return self.waypoints[-1].copy()
        # Find which segment we're in
        for i in range(len(self.times) - 1):
            if self.times[i] <= t < self.times[i + 1]:
                # Linear interpolation
                frac = (t - self.times[i]) / (self.times[i + 1] - self.times[i])
                return ((1 - frac) * self.waypoints[i]
                        + frac * self.waypoints[i + 1])
        return self.waypoints[-1].copy()


def ramp_drift(d=2, T=6000):
    """
    Same waypoints as the abrupt drift, but with linear ramps between them.
    This lets you directly compare "sharp" vs "smooth" versions of the
    same overall trajectory.
    """
    waypoints = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, -1.0]),
    ]
    times = [0, 1500, 3000, 4500]
    return PiecewiseLinearDrift(d=d, waypoints=waypoints, times=times)
