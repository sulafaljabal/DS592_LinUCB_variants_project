"""
algorithms/d_linucb_dynamic.py — Dynamic LinUCB (dLinUCB)
==========================================================
Reference: Wu, Iyer, Wang (SIGIR 2018)
    "Learning Contextual Bandits in a Non-stationary Environment"

As implemented in: YRussac/WeightedLinearBandits (dLinUCB_class.py)

Key idea: A master algorithm maintains a pool of LinUCB "slave" models.
Each slave is a standard LinUCB that was initialized at some point in time.
The master tracks the "badness" of each slave — defined as the number of
times in the last τ steps where the prediction error exceeded the confidence
bound. When a slave's badness exceeds a threshold, it is discarded. When a
change is detected (all slaves are bad), a new slave is created.

The master selects which slave to use via the Lower Confidence Bound (LCB)
of badness: it picks the slave whose badness is most confidently low.

Parameters:
    - tau (τ): window for badness estimation
    - delta_1 (δ₁): threshold for prediction error (controls false alarm rate)
    - delta_2 (δ₂): threshold for badness detection

This is fundamentally different from D-LinUCB and SW-LinUCB:
    - D-LinUCB / SW-LinUCB: passively adapt by forgetting old data
    - dLinUCB: actively detects change points and restarts estimation
"""

import numpy as np
from math import log, sqrt
from numpy.linalg import pinv
from algorithms.base import BanditAlgorithm


class _SlaveLinUCB:
    """
    A single LinUCB slave model with its own V, b, and θ̂.
    Created at a specific timestep and runs standard LinUCB from that point.
    """

    def __init__(self, d, lambda_reg, delta, sigma_noise, S, creation_time):
        self.d = d
        self.lambda_reg = lambda_reg
        self.delta = delta
        self.sigma_noise = sigma_noise
        self.S = S
        self.creation_time = creation_time

        self.V = lambda_reg * np.eye(d)
        self.b = np.zeros(d)
        self.V_inv = (1.0 / lambda_reg) * np.eye(d)
        self.theta_hat = np.zeros(d)
        self.t_local = 0  # number of updates this slave has seen

    def compute_beta(self):
        """Standard LinUCB confidence width."""
        if self.t_local == 0:
            return self.S + self.sigma_noise * sqrt(2 * log(1.0 / self.delta))
        return (sqrt(self.lambda_reg) * self.S
                + self.sigma_noise * sqrt(
                    2 * log(1.0 / self.delta)
                    + self.d * log(1.0 + self.t_local / (self.lambda_reg * self.d))
                ))

    def get_ucb(self, action):
        """Return UCB value for a single action vector."""
        beta = self.compute_beta()
        mean = action @ self.theta_hat
        bonus = beta * sqrt(action @ self.V_inv @ action)
        return mean + bonus

    def get_prediction_and_width(self, action):
        """Return (predicted reward, confidence width) for change detection."""
        beta = self.compute_beta()
        mean = action @ self.theta_hat
        width = beta * sqrt(action @ self.V_inv @ action)
        return mean, width

    def update(self, action, reward):
        """Standard LinUCB update (no discounting)."""
        self.t_local += 1
        aat = np.outer(action, action)
        self.V += aat
        self.b += reward * action
        self.V_inv = pinv(self.V)
        self.theta_hat = self.V_inv @ self.b


class DynamicLinUCB(BanditAlgorithm):
    """
    dLinUCB: Change-point detection with master/slave LinUCB architecture.

    The master maintains a pool of LinUCB slaves and selects the one with
    the lowest confidence bound on "badness" (prediction error frequency).
    """

    def __init__(self, d, tau=200, delta_1=0.1, delta_2=0.1,
                 lambda_reg=1.0, delta=0.01, alpha=1.0,
                 sigma_noise=1.0, S=1.0):
        """
        Parameters
        ----------
        d : int
            Dimension.
        tau : int
            Window for badness estimation (how far back to check errors).
        delta_1 : float
            Threshold for declaring a prediction "bad". If the prediction
            error exceeds the confidence bound scaled by delta_1, that
            timestep counts toward badness.
        delta_2 : float
            Threshold for discarding a slave. If a slave's badness LCB
            exceeds delta_2, it is removed from the pool.
        lambda_reg : float
            Regularization λ for slave LinUCBs.
        delta : float
            Confidence δ for slave LinUCBs.
        alpha : float
            Exploration multiplier.
        sigma_noise : float
            Sub-Gaussian noise parameter.
        S : float
            Bound on ‖θ*‖.
        """
        super().__init__(d, lambda_reg, delta, name=f"dLinUCB(τ={tau})")
        self.tau = tau
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.alpha = alpha
        self.sigma_noise = sigma_noise
        self.S = S
        self.reset()

    def reset(self):
        self.t = 0
        # Create the initial slave at t=0
        initial_slave = _SlaveLinUCB(
            self.d, self.lambda_reg, self.delta,
            self.sigma_noise, self.S, creation_time=0
        )
        self.slaves = [initial_slave]
        # Badness tracking: for each slave, store recent error indicators
        # error_history[i] is a list of 0/1 values (was prediction "bad"?)
        self.error_histories = [[]]
        # Track which slave was selected (for the theta_hat property)
        self.active_slave_idx = 0
        self._theta_hat = np.zeros(self.d)

    @property
    def theta_hat(self):
        return self._theta_hat

    @theta_hat.setter
    def theta_hat(self, value):
        self._theta_hat = value

    def _compute_badness(self, slave_idx):
        """
        Compute the badness and its LCB for a slave.

        Badness = (number of "bad" predictions in last τ steps) / τ
        LCB = badness - confidence_width

        Returns (badness_mean, badness_lcb)
        """
        history = self.error_histories[slave_idx]
        if len(history) == 0:
            return 0.0, 0.0

        # Use only last τ entries
        recent = history[-self.tau:]
        n = len(recent)
        badness_mean = sum(recent) / n

        # Hoeffding-style confidence interval for the mean
        if n > 0:
            width = sqrt(log(2.0 / self.delta_1) / (2.0 * n))
        else:
            width = 1.0

        badness_lcb = badness_mean - width
        return badness_mean, badness_lcb

    def select_action(self, actions):
        """
        Select an action using the active slave (the one with lowest
        badness LCB).
        """
        # First, select the best slave via LCB of badness
        best_slave_idx = 0
        best_lcb = float('inf')

        for i in range(len(self.slaves)):
            _, lcb = self._compute_badness(i)
            if lcb < best_lcb:
                best_lcb = lcb
                best_slave_idx = i

        self.active_slave_idx = best_slave_idx
        slave = self.slaves[best_slave_idx]
        self._theta_hat = slave.theta_hat.copy()

        # Use the selected slave's UCB to pick an action
        n_actions = actions.shape[0]
        ucb_values = np.zeros(n_actions)
        beta = slave.compute_beta()

        for i in range(n_actions):
            a = actions[i]
            mean = a @ slave.theta_hat
            bonus = self.alpha * beta * sqrt(a @ slave.V_inv @ a)
            ucb_values[i] = mean + bonus

        # Tie-breaking matching author's lexsort approach
        mixer = np.random.random(ucb_values.size)
        ucb_indices = list(np.lexsort((mixer, ucb_values)))
        return ucb_indices[-1]

    def update(self, action, reward):
        """
        Update all admissible slaves and track badness.

        Steps:
        1. For each slave, check if the prediction error exceeds the
           confidence bound → record as "bad" (1) or "good" (0).
        2. Discard slaves whose badness LCB exceeds delta_2.
        3. If no slaves remain (all were bad), create a new one.
        4. Update all remaining slaves with the new data point.
        """
        self.t += 1

        # Step 1: Evaluate all slaves and track errors
        slaves_to_remove = []
        for i, slave in enumerate(self.slaves):
            # Check prediction quality
            predicted, width = slave.get_prediction_and_width(action)
            error = abs(reward - predicted)

            # Is the prediction "bad"? (error exceeds confidence bound)
            is_bad = 1 if error > width else 0
            self.error_histories[i].append(is_bad)

            # Keep only last τ entries to bound memory
            if len(self.error_histories[i]) > self.tau:
                self.error_histories[i] = self.error_histories[i][-self.tau:]

        # Step 2: Check badness and mark slaves for removal
        for i in range(len(self.slaves)):
            badness_mean, badness_lcb = self._compute_badness(i)
            if badness_lcb > self.delta_2:
                slaves_to_remove.append(i)

        # Remove bad slaves (in reverse order to preserve indices)
        for i in sorted(slaves_to_remove, reverse=True):
            del self.slaves[i]
            del self.error_histories[i]

        # Step 3: If pool is empty or all were removed, create a new slave
        if len(self.slaves) == 0 or len(slaves_to_remove) > 0:
            new_slave = _SlaveLinUCB(
                self.d, self.lambda_reg, self.delta,
                self.sigma_noise, self.S, creation_time=self.t
            )
            self.slaves.append(new_slave)
            self.error_histories.append([])

        # Step 4: Update all remaining slaves
        for slave in self.slaves:
            slave.update(action, reward)

        # Update theta_hat from the active slave
        if self.active_slave_idx < len(self.slaves):
            self._theta_hat = self.slaves[self.active_slave_idx].theta_hat.copy()
        else:
            self._theta_hat = self.slaves[-1].theta_hat.copy()
