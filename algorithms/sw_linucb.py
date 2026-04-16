"""
algorithms/sw_linucb.py — Sliding Window LinUCB (SW-LinUCB)
============================================================
Reference: Cheung, Simchi-Levi, Zhu (AISTATS 2019)
    "Learning to Optimize under Non-Stationarity"

Key idea: Only use the last τ observations for estimation.
    V_t = λI + Σ_{s=max(1,t-τ+1)}^{t} A_s A_s^⊤
    b_t = Σ_{s=max(1,t-τ+1)}^{t} A_s X_s

Unlike D-LinUCB which smoothly discounts, SW-LinUCB has a hard cutoff.
Must store the last τ (action, reward) pairs.
"""

import numpy as np
from math import log
from collections import deque
from numpy.linalg import pinv
from algorithms.base import BanditAlgorithm


class SWLinUCB(BanditAlgorithm):
    """Sliding Window LinUCB with window size τ."""

    def __init__(self, d, tau=200, lambda_reg=1.0, delta=0.01, alpha=1.0,
                 sigma_noise=1.0, S=1.0):
        """
        Parameters
        ----------
        d : int
            Dimension.
        tau : int
            Sliding window size τ.
        lambda_reg : float
            Regularization λ.
        delta : float
            Confidence δ.
        alpha : float
            Exploration multiplier.
        sigma_noise : float
            Sub-Gaussian noise parameter σ.
        S : float
            Upper bound on ‖θ*‖_2.
        """
        super().__init__(d, lambda_reg, delta, name=f"SW-LinUCB(τ={tau})")
        self.tau = tau
        self.alpha = alpha
        self.sigma_noise = sigma_noise
        self.S = S
        self.reset()

    def reset(self):
        self.t = 0
        self.history = deque(maxlen=self.tau)  # stores (action, reward) pairs
        self.V = self.lambda_reg * np.eye(self.d)
        self.b = np.zeros(self.d)
        self.V_inv = (1.0 / self.lambda_reg) * np.eye(self.d)
        self.theta_hat = np.zeros(self.d)

        # Precompute constants
        self.c_delta = 2 * log(1.0 / self.delta)
        self.const1 = np.sqrt(self.lambda_reg) * self.S

    def _rebuild(self):
        """Rebuild V and b from the current window. Called every update."""
        self.V = self.lambda_reg * np.eye(self.d)
        self.b = np.zeros(self.d)
        for (a, r) in self.history:
            self.V += np.outer(a, a)
            self.b += r * a
        # Use pinv matching the author's approach (SM not applicable
        # because the window removal is not a rank-1 update)
        self.V_inv = pinv(self.V)
        self.theta_hat = self.V_inv @ self.b

    def _compute_beta(self):
        """
        Confidence width for the sliding window.

        Uses effective time min(t, τ) in the log-determinant bound,
        matching the structure of the author's LinUCB beta.
        """
        effective_t = min(self.t, self.tau)
        beta = self.const1 + self.sigma_noise * np.sqrt(
            self.c_delta + self.d * log(1.0 + effective_t / (self.lambda_reg * self.d))
        )
        return beta

    def select_action(self, actions):
        """Pick action maximizing windowed UCB."""
        beta = self._compute_beta()
        n_actions = actions.shape[0]
        ucb_values = np.zeros(n_actions)

        for i in range(n_actions):
            a = actions[i]
            invcov_a = self.V_inv @ a
            mean = a @ self.theta_hat
            bonus = self.alpha * beta * np.sqrt(a @ invcov_a)
            ucb_values[i] = mean + bonus

        # Break ties randomly (matching author's lexsort approach)
        mixer = np.random.random(ucb_values.size)
        ucb_indices = list(np.lexsort((mixer, ucb_values)))
        chosen_arm = ucb_indices[-1]  # largest UCB
        return chosen_arm

    def update(self, action, reward):
        """
        Add (action, reward) to window. If window is full, oldest is
        automatically dropped (deque maxlen). Then rebuild V and b.

        NOTE: Rebuilding from scratch each step is O(τ d^2). For large τ
        you could do incremental add/remove. For τ ~ 200 and d ~ 2-5
        this is fine.
        """
        self.t += 1
        self.history.append((action.copy(), reward))
        self._rebuild()
