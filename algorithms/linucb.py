"""
algorithms/linucb.py — Standard LinUCB for stationary linear bandits
=====================================================================
Reference: Lattimore & Szepesvári, Ch. 19
    θ̂_t = V_t^{-1} Σ A_s X_s
    V_t = λI + Σ A_s A_s^⊤
    UCB_t(a) = ⟨θ̂_{t-1}, a⟩ + α * β_t * ‖a‖_{V_{t-1}^{-1}}

Sherman-Morrison implementation matches the author's repo (YRussac/WeightedLinearBandits).
When sm=True, V^{-1} is updated incrementally instead of recomputed from scratch.
When sm=False, pinv is used (safer but slower for large d).
"""

import numpy as np
from math import log
from numpy.linalg import pinv
from algorithms.base import BanditAlgorithm


class LinUCB(BanditAlgorithm):
    """Standard LinUCB (no discounting, no sliding window)."""

    def __init__(self, d, lambda_reg=1.0, delta=0.01, alpha=1.0,
                 sigma_noise=1.0, S=1.0, sm=True):
        """
        Parameters
        ----------
        d : int
            Dimension.
        lambda_reg : float
            Regularization λ.
        delta : float
            Confidence δ.
        alpha : float
            Exploration multiplier (set to 1.0 for theoretical value).
        sigma_noise : float
            Sub-Gaussian noise parameter σ.
        S : float
            Upper bound on ‖θ*‖_2.
        sm : bool
            If True, use Sherman-Morrison for incremental V^{-1} updates.
            If False, recompute pinv(V) each step (matching author's fallback).
        """
        super().__init__(d, lambda_reg, delta, name="LinUCB")
        self.alpha = alpha
        self.sigma_noise = sigma_noise
        self.S = S
        self.sm = sm
        self.reset()

    def reset(self):
        self.t = 0
        self.V = self.lambda_reg * np.eye(self.d)
        self.b = np.zeros(self.d)
        self.V_inv = (1.0 / self.lambda_reg) * np.eye(self.d)
        self.theta_hat = np.zeros(self.d)

        # Precompute constants
        self.c_delta = 2 * log(1.0 / self.delta)
        self.const1 = np.sqrt(self.lambda_reg) * self.S

    def _compute_beta(self):
        """
        Confidence width β_t matching the author's implementation.

        β_t = sqrt(λ) * S + σ * sqrt(2 * log(1/δ) + d * log(1 + t / (λ * d)))

        This is the standard closed-form upper bound on log(det(V_t)/det(V_0))
        when action norms are bounded by 1.
        """
        beta = self.const1 + self.sigma_noise * np.sqrt(
            self.c_delta + self.d * log(1.0 + self.t / (self.lambda_reg * self.d))
        )
        return beta

    def select_action(self, actions):
        """
        Pick action maximizing UCB = ⟨θ̂, a⟩ + α * β_t * ‖a‖_{V^{-1}}.
        """
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
        Update V, b, V^{-1}, and θ̂ after observing (action, reward).

        If sm=True, uses Sherman-Morrison formula matching the author:
            V^{-1}_{new} = V^{-1} - (V^{-1} a)(V^{-1} a)^⊤ / (1 + a^⊤ V^{-1} a)

        If sm=False, recomputes pinv(V) from scratch.
        """
        self.t += 1
        a = action
        aat = np.outer(a, a)

        self.V = self.V + aat
        self.b = self.b + reward * a

        if not self.sm:
            self.V_inv = pinv(self.V)
        else:
            # Sherman-Morrison (matching author's implementation exactly)
            # const = 1 / (1 + a^⊤ V^{-1} a)
            # const2 = V^{-1} a  (as column vector)
            # V^{-1}_{new} = V^{-1} - const * const2 @ const2^⊤
            a_col = a[:, np.newaxis]  # (d, 1)
            const = 1.0 / (1.0 + np.dot(a, self.V_inv @ a))
            const2 = self.V_inv @ a_col  # (d, 1)
            self.V_inv = self.V_inv - const * (const2 @ const2.T)

        self.theta_hat = self.V_inv @ self.b
