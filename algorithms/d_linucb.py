"""
algorithms/d_linucb.py — Discounted LinUCB (D-LinUCB)
======================================================
Reference: Russac, Vernade, Cappé (NeurIPS 2019)
    "Weighted Linear Bandits for Non-Stationary Environments"

Key idea: Use exponentially discounted least-squares with discount γ.
    V_t = λI + Σ_{s=1}^{t} γ^{t-s} A_s A_s^⊤
    b_t = Σ_{s=1}^{t} γ^{t-s} A_s X_s
    θ̂_t = V_t^{-1} b_t

The discount factor γ ∈ (0,1) controls how fast old data is forgotten.
    γ close to 1  → slow forgetting (like standard LinUCB)
    γ close to 0  → fast forgetting (only recent data matters)
"""

import numpy as np
from algorithms.base import BanditAlgorithm


class DLinUCB(BanditAlgorithm):
    """Discounted LinUCB with exponential weights."""

    def __init__(self, d, gamma=0.99, lambda_reg=1.0, delta=0.01,
                 sigma_noise=1.0, S=1.0):
        """
        Parameters
        ----------
        d : int
            Dimension.
        gamma : float
            Discount factor γ ∈ (0, 1).
        lambda_reg : float
            Regularization λ.
        delta : float
            Confidence δ.
        sigma_noise : float
            Sub-Gaussian noise parameter σ.
        S : float
            Upper bound on ‖θ*‖_2.
        """
        super().__init__(d, lambda_reg, delta, name=f"D-LinUCB(γ={gamma})")
        self.gamma = gamma
        self.sigma_noise = sigma_noise
        self.S = S
        self.reset()

    def reset(self):
        self.t = 0
        self.V = self.lambda_reg * np.eye(self.d)
        self.b = np.zeros(self.d)
        self.V_inv = np.eye(self.d) / self.lambda_reg
        self.theta_hat = np.zeros(self.d)

    def _compute_beta(self):
        """
        Confidence width for D-LinUCB.

        From Theorem 2 of Russac et al. (2019), adapted for the
        discounted setting. This is an approximation — the exact
        expression involves the effective dimension under discounting.
        """
        # Effective number of samples under discounting
        if self.gamma < 1.0:
            effective_t = (1 - self.gamma ** self.t) / (1 - self.gamma)
        else:
            effective_t = self.t

        log_det_ratio = np.log(
            max(np.linalg.det(self.V) / (self.lambda_reg ** self.d), 1.0)
        )
        beta = (self.sigma_noise
                * np.sqrt(2 * np.log(1.0 / self.delta) + log_det_ratio)
                + np.sqrt(self.lambda_reg) * self.S)
        return beta

    def select_action(self, actions):
        """Pick action maximizing discounted UCB."""
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
        Discounted update: multiply existing V and b by γ, then add new data.

        V_{t} = γ * V_{t-1} + a_t a_t^⊤ + (1 - γ) * λI
              (the (1-γ)λI term keeps regularization from decaying)

        b_{t} = γ * b_{t-1} + r_t * a_t
        """
        self.t += 1
        a = action

        # Discount existing statistics
        self.V = self.gamma * self.V + np.outer(a, a) + (1 - self.gamma) * self.lambda_reg * np.eye(self.d)
        self.b = self.gamma * self.b + reward * a

        # Recompute inverse (for modest d, this is fine; for large d use Sherman-Morrison)
        # NOTE: For d <= 10, direct inversion is fast and numerically stable.
        # For larger d, consider rank-1 update on the discounted matrix.
        self.V_inv = np.linalg.inv(self.V)
        self.theta_hat = self.V_inv @ self.b
