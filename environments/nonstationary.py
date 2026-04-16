"""
environments/nonstationary.py — Non-stationary linear bandit environment
=========================================================================
This is the simulation engine. It:
  1. Generates action sets each round
  2. Uses a DriftFunction to get the current θ*_t
  3. Generates rewards as X_t = ⟨θ*_t, A_t⟩ + η_t
  4. Tracks regret against the best action at each timestep
"""

import numpy as np


class NonStationaryLinearBandit:
    """Non-stationary linear bandit environment with pluggable drift."""

    def __init__(self, d, drift_fn, n_actions=10, sigma_noise=1.0,
                 fixed_actions=None):
        """
        Parameters
        ----------
        d : int
            Dimension.
        drift_fn : DriftFunction
            Callable that returns θ*_t given timestep t.
        n_actions : int
            Number of actions available each round.
        sigma_noise : float
            Standard deviation of Gaussian noise.
        fixed_actions : np.ndarray of shape (n_actions, d) or None
            If provided, use the same action set every round.
            If None, sample new actions each round from the unit sphere.
        """
        self.d = d
        self.drift_fn = drift_fn
        self.n_actions = n_actions
        self.sigma_noise = sigma_noise
        self.fixed_actions = fixed_actions

    def get_actions(self, t, rng):
        """Return the action set for round t."""
        if self.fixed_actions is not None:
            return self.fixed_actions.copy()
        # Sample random unit vectors
        actions = rng.randn(self.n_actions, self.d)
        norms = np.linalg.norm(actions, axis=1, keepdims=True)
        actions = actions / norms
        return actions

    def get_reward(self, t, action, rng):
        """
        Generate reward for playing `action` at timestep t.

        Returns
        -------
        reward : float
            Observed (noisy) reward.
        expected_reward : float
            True mean reward ⟨θ*_t, action⟩ (for regret computation).
        """
        theta_star = self.drift_fn(t)
        expected = theta_star @ action
        noise = rng.randn() * self.sigma_noise
        return expected + noise, expected

    def best_expected_reward(self, t, actions):
        """
        Return the best achievable expected reward at timestep t.

        Parameters
        ----------
        t : int
            Timestep.
        actions : np.ndarray of shape (n_actions, d)
            Available actions.

        Returns
        -------
        float
            max_a ⟨θ*_t, a⟩
        """
        theta_star = self.drift_fn(t)
        expected_rewards = actions @ theta_star
        return np.max(expected_rewards)


def run_experiment(algorithm, environment, T, n_simulations=100, seed=592):
    """
    Run a full Monte Carlo experiment.

    Parameters
    ----------
    algorithm : BanditAlgorithm
        The algorithm to evaluate (will be reset each simulation).
    environment : NonStationaryLinearBandit
        The environment to interact with.
    T : int
        Time horizon.
    n_simulations : int
        Number of independent runs.
    seed : int
        Base random seed.

    Returns
    -------
    dict with keys:
        'cumulative_regret' : np.ndarray of shape (n_simulations, T)
            Cumulative regret trajectory for each simulation.
        'mean_regret' : np.ndarray of shape (T,)
            Mean cumulative regret across simulations.
        'std_regret' : np.ndarray of shape (T,)
            Std of cumulative regret across simulations.
        'theta_estimates' : np.ndarray of shape (n_simulations, T, d)
            Parameter estimates over time (for diagnostic plots).
    """
    d = algorithm.d
    all_regret = np.zeros((n_simulations, T))
    all_theta_hat = np.zeros((n_simulations, T, d))

    for sim in range(n_simulations):
        rng = np.random.RandomState(seed + sim)
        algorithm.reset()

        cum_regret = 0.0
        for t in range(1, T + 1):
            # 1. Get available actions
            actions = environment.get_actions(t, rng)

            # 2. Algorithm selects an action
            action_idx = algorithm.select_action(actions)
            chosen_action = actions[action_idx]

            # 3. Environment generates reward
            reward, expected = environment.get_reward(t, chosen_action, rng)

            # 4. Compute instantaneous regret
            best_reward = environment.best_expected_reward(t, actions)
            instant_regret = best_reward - expected
            cum_regret += instant_regret

            # 5. Algorithm updates
            algorithm.update(chosen_action, reward)

            # 6. Record
            all_regret[sim, t - 1] = cum_regret
            all_theta_hat[sim, t - 1] = algorithm.theta_hat.copy()

        if (sim + 1) % 10 == 0:
            print(f"  {algorithm.name}: simulation {sim + 1}/{n_simulations}, "
                  f"final regret = {cum_regret:.1f}")

    return {
        'cumulative_regret': all_regret,
        'mean_regret': np.mean(all_regret, axis=0),
        'std_regret': np.std(all_regret, axis=0),
        'theta_estimates': all_theta_hat,
    }
