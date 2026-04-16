"""
drift_functions/base.py — Abstract interface for drift functions
================================================================
A drift function defines how θ* evolves over time.

    θ*_t = drift(t)

This is the key interface between the algorithm person and the
drift-function person. As long as your drift function implements
this interface, it plugs right into the environment.
"""

from abc import ABC, abstractmethod
import numpy as np


class DriftFunction(ABC):
    """Base class for all drift functions."""

    def __init__(self, d, name="BaseDrift"):
        """
        Parameters
        ----------
        d : int
            Dimension of θ*.
        name : str
            Display name for plots.
        """
        self.d = d
        self.name = name

    @abstractmethod
    def __call__(self, t):
        """
        Return θ* at timestep t.

        Parameters
        ----------
        t : int
            Current timestep (1-indexed).

        Returns
        -------
        np.ndarray of shape (d,)
            The true parameter vector at time t.
        """
        pass

    def total_variation(self, T):
        """
        Compute B_T = Σ_{t=1}^{T-1} ‖θ*_{t+1} - θ*_t‖_2.

        This is the non-stationarity measure from the paper.
        """
        bv = 0.0
        prev = self(1)
        for t in range(2, T + 1):
            curr = self(t)
            bv += np.linalg.norm(curr - prev)
            prev = curr
        return bv

    def __repr__(self):
        return f"{self.name}(d={self.d})"
