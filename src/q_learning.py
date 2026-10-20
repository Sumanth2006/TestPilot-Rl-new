"""
TestPilot-RL: Q-Learning Agent with Linear Function Approximation.

This module implements TWO complementary agents:
1. LinearQLearningAgent - Uses per-test feature vectors learned via gradient descent.
2. TabularQLearningAgent - Legacy tabular agent for compatibility and ablation.

Features use observable signals only. Hidden bug information must not leak
into the agent's decision-making process.
"""

import os
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any


# ─────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────

NUM_FEATURES = 8


def extract_test_features(
    action: int,
    env: Any,
    test_history: Dict[int, float],
) -> np.ndarray:
    """
    Construct an 8-dimensional feature vector for a candidate test.

    Features:
      0. Historical failure rate
      1. Failure rate per cost
      2. New coverage features, normalized
      3. Coverage efficiency per cost
      4. Test cost / remaining budget
      5. Remaining budget / total budget
      6. Coverage progress
      7. Whether recent tests found bugs
    """
    build = env.build
    cost = max(1e-4, float(build.costs[action]))

    fail_rate = float(test_history.get(action, 0.05))
    fail_per_cost = fail_rate / cost

    total_feats = max(1, build.total_features)
    test_feats = build.coverage_map.get(action, set())
    new_feats = len(test_feats - env.covered_features)

    new_cov_norm = new_feats / total_feats
    cov_per_cost = new_cov_norm / cost

    rem_budget = max(1e-5, float(env.remaining_budget))
    total_budget = max(1e-5, float(env.total_budget))

    cost_ratio = min(1.0, cost / rem_budget)
    budget_remaining = min(1.0, rem_budget / total_budget)
    coverage_progress = min(
        1.0,
        len(env.covered_features) / total_feats,
    )

    recent_bugs = sum(env.recent_outcomes[-3:]) if env.recent_outcomes else 0
    recent_success = float(recent_bugs > 0)

    return np.array(
        [
            fail_rate,
            min(fail_per_cost, 10.0) / 10.0,
            new_cov_norm,
            min(cov_per_cost, 10.0) / 10.0,
            cost_ratio,
            budget_remaining,
            coverage_progress,
            recent_success,
        ],
        dtype=np.float64,
    )


# ─────────────────────────────────────────────────────────
# Prioritized Experience Replay Buffer
# ─────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """Replay buffer that samples transitions by their TD-error priority."""

    def __init__(self, capacity: int = 20000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)
        self.max_priority = 1.0

    def push(self, transition: tuple):
        self.buffer.append(transition)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size: int, beta: float = 0.4):
        if not self.buffer:
            return [], [], []

        priorities = np.asarray(self.priorities, dtype=np.float64)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        n = len(self.buffer)
        indices = np.random.choice(
            n,
            size=min(batch_size, n),
            p=probabilities,
            replace=False,
        )

        transitions = [self.buffer[i] for i in indices]
        weights = (n * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return transitions, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = max(abs(float(error)), 1e-6)
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────
# Linear Q-Learning Agent
# ─────────────────────────────────────────────────────────

class LinearQLearningAgent:
    """
    Linear function approximation:

        Q(s, a) = w · φ(s, a)

    Uses experience replay and a target weight vector.

    IMPORTANT:
    The features for the action just executed should be captured before
    env.step(action). Pass them to update() using action_features.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.997,
        replay_capacity: int = 20000,
        batch_size: int = 64,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        beta_anneal_steps: int = 10000,
        shaping_coef: float = 2.0,
        seed: Optional[int] = None,
        test_history: Optional[Dict[int, float]] = None,
        num_actions: int = 0,
        budget_bins: int = 5,
        coverage_bins: int = 5,
        history_bins: int = 3,
        w_fail: float = 0.0,
        w_cov: float = 0.0,
    ):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.shaping_coef = float(shaping_coef)

        self.test_history: Dict[int, float] = test_history or {}

        self.rng = np.random.default_rng(seed)

        scale = np.sqrt(2.0 / NUM_FEATURES)
        self.w = self.rng.normal(0.0, scale, NUM_FEATURES)
        self.w_target = self.w.copy()

        self.target_update_freq = 200
        self._step_count = 0

        self.replay = PrioritizedReplayBuffer(
            capacity=replay_capacity,
            alpha=per_alpha,
        )

        self.per_beta = float(per_beta_start)
        self.per_beta_end = float(per_beta_end)
        self.per_beta_step = (
            (per_beta_end - per_beta_start) / max(1, beta_anneal_steps)
        )

        self.num_actions = num_actions

    def set_test_history(self, test_history: Dict[int, float]):
        self.test_history = test_history

    def _q_value(
        self,
        features: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        selected_weights = self.w if weights is None else weights
        return float(np.dot(selected_weights, features))

    def get_action_value(self, state, action: int, env: Any = None) -> float:
        """Public interface compatible with the tabular agent."""
        if env is None:
            return 0.0

        features = extract_test_features(action, env, self.test_history)
        return self._q_value(features)

    def select_action(
        self,
        state,
        legal_actions: List[int],
        explore: bool = True,
        env: Any = None,
    ) -> int:
        if not legal_actions:
            raise ValueError("No legal actions available.")

        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal_actions))

        if env is not None:
            scores = [
                self._q_value(
                    extract_test_features(action, env, self.test_history)
                )
                for action in legal_actions
            ]
        else:
            scores = [0.0] * len(legal_actions)

        best_score = max(scores)
        best_actions = [
            action
            for action, score in zip(legal_actions, scores)
            if np.isclose(score, best_score, atol=1e-8)
        ]

        return int(self.rng.choice(best_actions))

    def _shaped_reward(
        self,
        reward: float,
        action: int,
        env: Any,
    ) -> float:
        """
        Add a failure-rate-per-cost shaping bonus.

        The bonus is annealed with epsilon. This is an auxiliary training
        signal, not a guarantee that the learned policy will beat a baseline.
        """
        fail_rate = float(self.test_history.get(action, 0.05))
        cost = max(1e-4, float(env.build.costs[action]))

        shaping_bonus = self.shaping_coef * (fail_rate / cost)

        epsilon_fraction = max(
            0.0,
            (self.epsilon - self.epsilon_min)
            / max(1e-9, 1.0 - self.epsilon_min),
        )

        return float(reward + epsilon_fraction * shaping_bonus)

    def update(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        next_legal_actions: List[int],
        terminated: bool,
        env: Any = None,
        action_features: Optional[np.ndarray] = None,
    ) -> float:
        """
        Store a transition and update the model using replay.

        Parameters
        ----------
        action_features:
            Features for the chosen action captured BEFORE env.step(action).
            Pass this argument from the training loop. If omitted, features
            are extracted from the environment as received for compatibility;
            that fallback may describe the wrong state if env has advanced.
        """
        if env is None:
            return 0.0

        shaped_reward = self._shaped_reward(reward, action, env)

        if action_features is None:
            features = extract_test_features(
                action,
                env,
                self.test_history,
            )
        else:
            features = np.asarray(action_features, dtype=np.float64).copy()

            if features.shape != (NUM_FEATURES,):
                raise ValueError(
                    "action_features must have shape "
                    f"({NUM_FEATURES},), got {features.shape}"
                )

        # env is normally in the next state here, so next-action features
        # are computed from the post-step environment.
        next_features = []

        if not terminated and next_legal_actions:
            next_features = [
                extract_test_features(
                    next_action,
                    env,
                    self.test_history,
                )
                for next_action in next_legal_actions
            ]

        self.replay.push(
            (
                features,
                shaped_reward,
                next_features,
                bool(terminated),
            )
        )

        mean_td_error = 0.0

        if len(self.replay) >= self.batch_size:
            mean_td_error = self._update_batch()

        self._step_count += 1

        if self._step_count % self.target_update_freq == 0:
            self.w_target = self.w.copy()

        return mean_td_error

    def _update_batch(self) -> float:
        """Perform one mini-batch gradient update."""
        transitions, indices, importance_weights = self.replay.sample(
            self.batch_size,
            beta=self.per_beta,
        )

        if not transitions:
            return 0.0

        self.per_beta = min(
            self.per_beta_end,
            self.per_beta + self.per_beta_step,
        )

        td_errors = []
        gradient_accumulator = np.zeros(NUM_FEATURES, dtype=np.float64)

        for transition, importance_weight in zip(
            transitions,
            importance_weights,
        ):
            features, reward, next_features, terminated = transition

            current_q = self._q_value(features)

            if terminated or not next_features:
                target = reward
            else:
                # Double Q-learning: select using online weights and evaluate
                # using target weights.
                next_values = [
                    self._q_value(next_feature, self.w)
                    for next_feature in next_features
                ]

                best_index = int(np.argmax(next_values))
                next_q = self._q_value(
                    next_features[best_index],
                    self.w_target,
                )

                target = reward + self.gamma * next_q

            td_error = float(target - current_q)
            td_errors.append(abs(td_error))

            gradient_accumulator += (
                importance_weight * td_error * features
            )

        self.w += (
            self.alpha / max(1, len(transitions))
        ) * gradient_accumulator

        self.w = np.clip(self.w, -50.0, 50.0)

        self.replay.update_priorities(indices, td_errors)

        return float(np.mean(td_errors))

    def decay_epsilon(self) -> float:
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )
        return self.epsilon

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        np.save(
            filepath,
            {
                "w": self.w,
                "w_target": self.w_target,
                "epsilon": self.epsilon,
                "gamma": self.gamma,
                "alpha": self.alpha,
                "agent_type": "linear",
            },
            allow_pickle=True,
        )

    def load(self, filepath: str):
        data = np.load(filepath, allow_pickle=True).item()

        if data.get("agent_type") == "linear":
            self.w = np.asarray(data["w"], dtype=np.float64)
            self.w_target = np.asarray(
                data.get("w_target", self.w.copy()),
                dtype=np.float64,
            )
            self.epsilon = float(
                data.get("epsilon", self.epsilon_min)
            )
            self.gamma = float(data.get("gamma", self.gamma))
            self.alpha = float(data.get("alpha", self.alpha))


# ─────────────────────────────────────────────────────────
# Legacy Tabular Q-Learning Agent
# ─────────────────────────────────────────────────────────

class TabularQLearningAgent:
    """
    Tabular Q-learning agent retained for compatibility and ablation.

    Q-table dimensions:
        [budget_bin, coverage_bin, history_bin, action]
    """

    def __init__(
        self,
        num_actions: int = 30,
        budget_bins: int = 5,
        coverage_bins: int = 5,
        history_bins: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
        test_history: Optional[Dict[int, float]] = None,
        w_fail: float = 12.0,
        w_cov: float = 10.0,
    ):
        self.num_actions = int(num_actions)
        self.budget_bins = int(budget_bins)
        self.coverage_bins = int(coverage_bins)
        self.history_bins = int(history_bins)

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.test_history = test_history or {}
        self.w_fail = float(w_fail)
        self.w_cov = float(w_cov)

        self.rng = np.random.default_rng(seed)

        self.q_table = np.zeros(
            (
                self.budget_bins,
                self.coverage_bins,
                self.history_bins,
                self.num_actions,
            ),
            dtype=np.float64,
        )

    def set_test_history(self, test_history: Dict[int, float]):
        self.test_history = test_history

    def get_action_value(
        self,
        state: Tuple[int, int, int],
        action: int,
        env: Any = None,
    ) -> float:
        budget_bin, coverage_bin, history_bin = state
        action_index = min(action, self.num_actions - 1)

        base_q = self.q_table[
            budget_bin,
            coverage_bin,
            history_bin,
            action_index,
        ]

        if env is not None and env.build is not None:
            build = env.build
            cost = max(1e-4, float(build.costs[action]))
            historical_failure_rate = float(
                self.test_history.get(action, 0.05)
            )

            new_features = len(
                build.coverage_map.get(action, set())
                - env.covered_features
            )

            failure_efficiency = (
                historical_failure_rate + 1e-4
            ) / cost

            coverage_efficiency = new_features / cost

            order_priority = 1.0 - (
                action / max(1, build.num_tests)
            )

            return float(
                base_q
                + self.w_fail * failure_efficiency
                + self.w_cov * coverage_efficiency
                + 0.5 * order_priority
            )

        return float(base_q)

    def select_action(
        self,
        state: Tuple[int, int, int],
        legal_actions: List[int],
        explore: bool = True,
        env: Any = None,
    ) -> int:
        if not legal_actions:
            raise ValueError(
                "No legal actions available to select from."
            )

        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal_actions))

        q_values = [
            self.get_action_value(state, action, env=env)
            for action in legal_actions
        ]

        max_q = max(q_values)

        best_actions = [
            action
            for action, q_value in zip(legal_actions, q_values)
            if np.isclose(q_value, max_q, atol=1e-8)
        ]

        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: Tuple[int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int],
        next_legal_actions: List[int],
        terminated: bool,
        env: Any = None,
    ) -> float:
        budget_bin, coverage_bin, history_bin = state
        action_index = min(action, self.num_actions - 1)

        current_q = self.q_table[
            budget_bin,
            coverage_bin,
            history_bin,
            action_index,
        ]

        if terminated or not next_legal_actions:
            target = reward
        else:
            next_budget, next_coverage, next_history = next_state

            next_q_values = [
                self.q_table[
                    next_budget,
                    next_coverage,
                    next_history,
                    min(next_action, self.num_actions - 1),
                ]
                for next_action in next_legal_actions
            ]

            target = reward + self.gamma * max(next_q_values)

        td_error = float(target - current_q)

        self.q_table[
            budget_bin,
            coverage_bin,
            history_bin,
            action_index,
        ] += self.alpha * td_error

        return td_error

    def decay_epsilon(self) -> float:
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )
        return self.epsilon

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        np.save(
            filepath,
            {
                "q_table": self.q_table,
                "num_actions": self.num_actions,
                "budget_bins": self.budget_bins,
                "coverage_bins": self.coverage_bins,
                "history_bins": self.history_bins,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "agent_type": "tabular",
            },
            allow_pickle=True,
        )

    def load(self, filepath: str):
        data = np.load(filepath, allow_pickle=True).item()

        if data.get("agent_type", "tabular") == "tabular":
            self.q_table = data["q_table"]
            self.num_actions = data["num_actions"]
            self.budget_bins = data["budget_bins"]
            self.coverage_bins = data["coverage_bins"]
            self.history_bins = data["history_bins"]

            self.alpha = data.get("alpha", self.alpha)
            self.gamma = data.get("gamma", self.gamma)
            self.epsilon = data.get("epsilon", self.epsilon)