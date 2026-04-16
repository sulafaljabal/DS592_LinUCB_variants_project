"""
algorithms/base.py — Abstract base class for all bandit algorithms
===================================================================
Every algorithm must implement:
  - select_action(actions): given a set of action vectors, return index
  - update(action, reward): update internal state after observing reward
  - reset(): reset to initial state
"""

from abc import ABC, abstractmethod
import numpy as np


class BanditAlgorithm(ABC):
    """Base class for linear bandit algorithms."""

    def __init__(self, d, lambda_reg=1.0, delta=0.01, name="BaseAlgorithm"):
        """
        Parameters
        ----------
        d : int
            Dimension of action/parameter vectors.
        lambda_reg : float
            Regularization parameter λ.
        delta : float
            Confidence parameter δ (probability of failure).
        name : str
            Display name for plots/logs.
        """
        self.d = d
        self.lambda_reg = lambda_reg
        self.delta = delta
        self.name = name
        self.t = 0  # current timestep

    @abstractmethod
    def select_action(self, actions):
        """
        Choose an action from the available set.

        Parameters
        ----------
        actions : np.ndarray of shape (n_actions, d)
            Available action vectors for this round.

        Returns
        -------
        int
            Index of the chosen action.
        """
        pass

    @abstractmethod
    def update(self, action, reward):
        """
        Update internal state after observing (action, reward).

        Parameters
        ----------
        action : np.ndarray of shape (d,)
            The action vector that was played.
        reward : float
            The observed reward.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset algorithm to initial state (for new simulation run)."""
        pass

    def __repr__(self):
        return f"{self.name}(d={self.d})"
