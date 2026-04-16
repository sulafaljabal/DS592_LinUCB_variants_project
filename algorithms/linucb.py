"""
algorithms/linucb.py — Standard LinUCB for stationary linear bandits
=====================================================================
Reference: Lattimore & Szepesvári, Ch. 19
    θ̂_t = V_t^{-1} Σ A_s X_s
    V_t = λI + Σ A_s A_s^⊤
    UCB_t(a) = ⟨θ̂_{t-1}, a⟩ + β_t * ‖a‖_{V_{t-1}^{-1}}
"""

import numpy as np
from algorithms.base import BanditAlgorithm


class LinUCB(BanditAlgorithm):
    """Standard LinUCB (no discounting, no sliding window)."""

    def __init__(self, d, lambda_reg=1.0, delta=0.01, sigma_noise=1.0, S=1.0):
        """
        Parameters
        ----------
        d : int
            Dimension.
        lambda_reg : float
            Regularization λ.
        delta : float
            Confidence δ.
        sigma_noise : float
            Sub-Gaussian noise parameter σ.
        S : float
            Upper bound on ‖θ*‖_2.
        """
        super().__init__(d, lambda_reg, delta, name="LinUCB")
        self.sigma_noise = sigma_noise
        self.S = S
        self.reset()

    def reset(self):
        self.t = 0
        self.V = self.lambda_reg * np.eye(self.d)       # design matrix
        self.b = np.zeros(self.d)                         # reward-weighted sum
        self.V_inv = np.eye(self.d) / self.lambda_reg    # cached inverse
        self.theta_hat = np.zeros(self.d)                 # current estimate

    def _compute_beta(self):
        """Confidence width β_t from Theorem 20.5 (Abbasi-Yadkori et al.)."""
        # β_t = σ * sqrt(2 log(1/δ) + d log(1 + t/(λd))) + sqrt(λ) * S
        log_det_ratio = np.log(np.linalg.det(self.V) / (self.lambda_reg ** self.d))
        beta = (self.sigma_noise
                * np.sqrt(2 * np.log(1.0 / self.delta) + log_det_ratio)
                + np.sqrt(self.lambda_reg) * self.S)
        return beta

    def select_action(self, actions):
        """
        Pick action maximizing UCB = ⟨θ̂, a⟩ + β * ‖a‖_{V^{-1}}.

        Parameters
        ----------
        actions : np.ndarray of shape (n_actions, d)

        Returns
        -------
        int : index of chosen action
        """
        beta = self._compute_beta()
        n_actions = actions.shape[0]
        ucb_values = np.zeros(n_actions)

        for i in range(n_actions):
            a = actions[i]
            mean = a @ self.theta_hat
            bonus = beta * np.sqrt(a @ self.V_inv @ a)
            ucb_values[i] = mean + bonus

        # Break ties randomly
        max_val = np.max(ucb_values)
        candidates = np.where(np.isclose(ucb_values, max_val))[0]
        return np.random.choice(candidates)

    def update(self, action, reward):
        """
        Rank-1 update of V and θ̂ after observing (action, reward).

        Uses Sherman-Morrison for efficient V^{-1} update.
        """
        self.t += 1
        a = action

        # Sherman-Morrison: (V + aa^⊤)^{-1} = V^{-1} - (V^{-1}a)(a^⊤V^{-1}) / (1 + a^⊤V^{-1}a)
        Va = self.V_inv @ a
        denom = 1.0 + a @ Va
        self.V_inv -= np.outer(Va, Va) / denom

        self.V += np.outer(a, a)
        self.b += reward * a
        self.theta_hat = self.V_inv @ self.b
