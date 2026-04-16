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
from collections import deque
from algorithms.base import BanditAlgorithm


class SWLinUCB(BanditAlgorithm):
    """Sliding Window LinUCB with window size τ."""

    def __init__(self, d, tau=200, lambda_reg=1.0, delta=0.01,
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
        sigma_noise : float
            Sub-Gaussian noise parameter σ.
        S : float
            Upper bound on ‖θ*‖_2.
        """
        super().__init__(d, lambda_reg, delta, name=f"SW-LinUCB(τ={tau})")
        self.tau = tau
        self.sigma_noise = sigma_noise
        self.S = S
        self.reset()

    def reset(self):
        self.t = 0
        self.history = deque(maxlen=self.tau)  # stores (action, reward) pairs
        self.V = self.lambda_reg * np.eye(self.d)
        self.b = np.zeros(self.d)
        self.V_inv = np.eye(self.d) / self.lambda_reg
        self.theta_hat = np.zeros(self.d)

    def _rebuild(self):
        """Rebuild V and b from the current window. Called every update."""
        self.V = self.lambda_reg * np.eye(self.d)
        self.b = np.zeros(self.d)
        for (a, r) in self.history:
            self.V += np.outer(a, a)
            self.b += r * a
        self.V_inv = np.linalg.inv(self.V)
        self.theta_hat = self.V_inv @ self.b

    def _compute_beta(self):
        """Confidence width adapted for the sliding window."""
        effective_t = min(self.t, self.tau)
        log_det_ratio = np.log(
            max(np.linalg.det(self.V) / (self.lambda_reg ** self.d), 1.0)
        )
        beta = (self.sigma_noise
                * np.sqrt(2 * np.log(1.0 / self.delta) + log_det_ratio)
                + np.sqrt(self.lambda_reg) * self.S)
        return beta

    def select_action(self, actions):
        """Pick action maximizing windowed UCB."""
        beta = self._compute_beta()
        n_actions = actions.shape[0]
        ucb_values = np.zeros(n_actions)

        for i in range(n_actions):
            a = actions[i]
            mean = a @ self.theta_hat
            bonus = beta * np.sqrt(a @ self.V_inv @ a)
            ucb_values[i] = mean + bonus

        max_val = np.max(ucb_values)
        candidates = np.where(np.isclose(ucb_values, max_val))[0]
        return np.random.choice(candidates)

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
