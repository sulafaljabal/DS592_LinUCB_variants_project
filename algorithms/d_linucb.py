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

IMPORTANT: The UCB bonus uses a "sandwich" covariance V^{-1} V_sq V^{-1}
where V_sq is the γ²-discounted design matrix. This comes from the variance
of the weighted least-squares estimator (the weights appear squared in the
variance). See Theorem 2 of the paper.
"""

import numpy as np
from math import log
from numpy.linalg import pinv
from algorithms.base import BanditAlgorithm


class DLinUCB(BanditAlgorithm):
    """Discounted LinUCB with exponential weights."""

    def __init__(self, d, gamma=0.99, lambda_reg=1.0, delta=0.01,
                 alpha=1.0, sigma_noise=1.0, S=1.0):
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
        alpha : float
            Exploration multiplier (set to 1.0 for theoretical value,
            tune down for practical performance).
        sigma_noise : float
            Sub-Gaussian noise parameter σ.
        S : float
            Upper bound on ‖θ*‖_2.
        """
        super().__init__(d, lambda_reg, delta, name=f"D-LinUCB(γ={gamma})")
        self.gamma = gamma
        self.alpha = alpha
        self.sigma_noise = sigma_noise
        self.S = S
        self.reset()

    def reset(self):
        self.t = 0
        self.V = self.lambda_reg * np.eye(self.d)            # γ-discounted design matrix
        self.V_sq = self.lambda_reg * np.eye(self.d)          # γ²-discounted design matrix
        self.b = np.zeros(self.d)
        self.V_inv = (1.0 / self.lambda_reg) * np.eye(self.d)
        self.theta_hat = np.zeros(self.d)
        self.gamma2_t = 1.0   # tracks γ^{2t}, updated multiplicatively

        # Precompute constants
        self.c_delta = 2 * log(1.0 / self.delta)

    def _compute_beta(self):
        """
        Exact confidence width β_t from Theorem 2 of Russac et al. (2019).

        β_t = sqrt(λ) * S + σ * sqrt(
            2 * log(1/δ) + d * log(1 + (1 - γ^{2t}) / (d * λ * (1 - γ²)))
        )

        The log term captures the effective dimension under discounting:
        as t grows, γ^{2t} → 0 and the log term stabilizes at
        d * log(1 + 1/(d * λ * (1 - γ²))), which depends on γ.
        """
        const1 = np.sqrt(self.lambda_reg) * self.S

        # Avoid division by zero when γ = 1 (falls back to standard LinUCB)
        if self.gamma < 1.0:
            log_arg = 1.0 + (1.0 - self.gamma2_t) / (self.d * self.lambda_reg * (1.0 - self.gamma ** 2))
        else:
            log_arg = 1.0 + self.t / (self.lambda_reg * self.d)

        beta = const1 + self.sigma_noise * np.sqrt(
            self.c_delta + self.d * log(log_arg)
        )
        return beta

    def select_action(self, actions):
        """
        Pick action maximizing discounted UCB.

        UCB(a) = ⟨θ̂, a⟩ + α * β_t * sqrt(a^⊤ V^{-1} V_sq V^{-1} a)

        The sandwich form V^{-1} V_sq V^{-1} is the correct variance
        proxy for the discounted estimator. Using just V^{-1} would
        underestimate the variance.
        """
        beta = self._compute_beta()
        n_actions = actions.shape[0]
        ucb_values = np.zeros(n_actions)

        for i in range(n_actions):
            a = actions[i]
            # Sandwich: a^⊤ V^{-1} V_sq V^{-1} a
            invcov_a = self.V_inv @ self.V_sq @ self.V_inv @ a
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
        Discounted update of both V (γ-discounted) and V_sq (γ²-discounted).

        V_{t}    = γ   * V_{t-1}    + a_t a_t^⊤ + (1 - γ)   * λI
        V_sq_{t} = γ²  * V_sq_{t-1} + a_t a_t^⊤ + (1 - γ²)  * λI
        b_{t}    = γ   * b_{t-1}    + r_t * a_t

        Sherman-Morrison CANNOT be used here because the discount
        means it's not a simple rank-1 update of the previous inverse.
        We use pinv (matching the author's implementation).
        """
        self.t += 1
        a = action
        aat = np.outer(a, a)

        # Update γ^{2t} tracker
        self.gamma2_t *= self.gamma ** 2

        # γ-discounted design matrix (used for estimation)
        self.V = (self.gamma * self.V
                  + aat
                  + (1.0 - self.gamma) * self.lambda_reg * np.eye(self.d))

        # γ²-discounted design matrix (used for the variance sandwich)
        self.V_sq = (self.gamma ** 2 * self.V_sq
                     + aat
                     + (1.0 - self.gamma ** 2) * self.lambda_reg * np.eye(self.d))

        # Reward vector
        self.b = self.gamma * self.b + reward * a

        # Recompute inverse via pinv (matching author; SM not applicable)
        self.V_inv = pinv(self.V)
        self.theta_hat = self.V_inv @ self.b
