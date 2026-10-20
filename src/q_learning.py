"""
TestPilot-RL: Q-Learning Agent with Linear Function Approximation.

This module implements TWO complementary agents:
1. LinearQLearningAgent - Uses per-test feature vectors (failure_rate/cost,
   coverage_efficiency, cost_ratio, recency) learned via gradient descent.
   Generalizes across builds with different test sets. This is the primary agent.
2. TabularQLearningAgent - Legacy tabular agent (kept for compatibility and ablation).

The LinearQLearningAgent outperforms greedy baselines by learning a globally
transferable scoring function over observable per-test signals.

Key design decisions:
- Feature vector encodes *observable* signals only (no hidden bug leak).
- Prioritized experience replay (PER) re-uses high-TD-error transitions.
- Shaped reward: extra shaping bonus for tests that historically fail often
  (bridges Q-learning toward the greedy oracle at convergence).
- Double Q-learning update reduces overestimation bias.
"""

import os
import random
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any


# ─────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────

NUM_FEATURES = 8  # dimension of the per-test feature vector


def extract_test_features(
    action: int,
    env: Any,
    test_history: Dict[int, float],
) -> np.ndarray:
    """
    Constructs an 8-dimensional feature vector for a candidate test action.
    All features are observable at selection time (no information leakage).

    Features:
      0  fail_rate           – historical failure rate of this test [0,1]
      1  fail_per_cost       – fail_rate / cost (core greedy signal)
      2  new_cov_feats       – # new features this test would cover (normalized)
      3  cov_per_cost        – new_cov_feats / cost
      4  cost_ratio          – test cost / remaining budget [0,1]
      5  budget_remaining    – remaining_budget / total_budget [0,1]
      6  coverage_progress   – features already covered / total_features [0,1]
      7  recent_success      – 1 if last 3 tests found bugs (momentum signal)
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

    rem_budget = max(1e-5, env.remaining_budget)
    total_budget = max(1e-5, env.total_budget)
    cost_ratio = min(1.0, cost / rem_budget)
    budget_remaining = min(1.0, rem_budget / total_budget)

    coverage_progress = min(1.0, len(env.covered_features) / total_feats)

    recent_bugs = sum(env.recent_outcomes[-3:]) if env.recent_outcomes else 0
    recent_success = float(recent_bugs > 0)

    feat = np.array([
        fail_rate,
        min(fail_per_cost, 10.0) / 10.0,   # normalize to [0,1]
        new_cov_norm,
        min(cov_per_cost, 10.0) / 10.0,    # normalize to [0,1]
        cost_ratio,
        budget_remaining,
        coverage_progress,
        recent_success,
    ], dtype=np.float64)
    return feat


# ─────────────────────────────────────────────────────────
# Prioritized Experience Replay Buffer
# ─────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Prioritized experience replay that samples transitions
    proportional to their absolute TD-error.
    """

    def __init__(self, capacity: int = 20000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # priority exponent
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)
        self.max_priority = 1.0

    def push(self, transition: tuple):
        self.buffer.append(transition)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == 0:
            return [], [], []
        prios = np.array(self.priorities, dtype=np.float64)
        probs = prios ** self.alpha
        probs /= probs.sum()
        n = len(self.buffer)
        indices = np.random.choice(n, size=min(batch_size, n), p=probs, replace=False)
        transitions = [self.buffer[i] for i in indices]
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()
        return transitions, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            p = max(abs(float(err)), 1e-6)
            self.priorities[idx] = p
            if p > self.max_priority:
                self.max_priority = p

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────
# Linear Q-Learning Agent (primary agent)
# ─────────────────────────────────────────────────────────

class LinearQLearningAgent:
    """
    Q(s, a) = w · φ(s, a)  — linear function approximator.

    The weight vector w is learned via semi-gradient Q-learning with
    prioritized experience replay and double Q-learning updates.

    Because features encode per-test *observable* signals
    (failure rate / cost, coverage efficiency, budget ratio…),
    the agent generalizes across builds and test-suite sizes.
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
        # kept for interface compatibility
        num_actions: int = 0,
        budget_bins: int = 5,
        coverage_bins: int = 5,
        history_bins: int = 3,
        w_fail: float = 0.0,
        w_cov: float = 0.0,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.shaping_coef = shaping_coef

        self.test_history: Dict[int, float] = test_history or {}

        # Weight vector (one per feature dimension); use Xavier-style init
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / NUM_FEATURES)
        self.w = rng.normal(0, scale, NUM_FEATURES)
        # Target network weights (Double Q-learning)
        self.w_target = self.w.copy()
        self.target_update_freq = 200   # steps between target network syncs
        self._step_count = 0

        self.replay = PrioritizedReplayBuffer(capacity=replay_capacity, alpha=per_alpha)
        self.per_beta = per_beta_start
        self.per_beta_end = per_beta_end
        self.per_beta_step = (per_beta_end - per_beta_start) / max(1, beta_anneal_steps)

        self.rng = rng

        # Stats
        self.num_actions = num_actions  # kept for compat

    def set_test_history(self, test_history: Dict[int, float]):
        self.test_history = test_history

    def _q_value(self, feat: np.ndarray, weights: np.ndarray = None) -> float:
        """Computes Q-value as dot product of weight vector and feature vector."""
        w = weights if weights is not None else self.w
        return float(np.dot(w, feat))

    def get_action_value(self, state, action: int, env: Any = None) -> float:
        """Public interface matching TabularQLearningAgent for compatibility."""
        if env is None:
            return 0.0
        feat = extract_test_features(action, env, self.test_history)
        return self._q_value(feat)

    def select_action(
        self,
        state,
        legal_actions: List[int],
        explore: bool = True,
        env: Any = None,
    ) -> int:
        if not legal_actions:
            raise ValueError("No legal actions available.")

        if explore and (self.rng.random() < self.epsilon):
            return int(self.rng.choice(legal_actions))

        # Exploitation: score each legal action using Q-function
        if env is not None:
            scores = [
                self._q_value(extract_test_features(a, env, self.test_history))
                for a in legal_actions
            ]
        else:
            scores = [0.0] * len(legal_actions)

        best_score = max(scores)
        best_actions = [
            a for a, s in zip(legal_actions, scores)
            if np.isclose(s, best_score, atol=1e-8)
        ]
        return int(self.rng.choice(best_actions))

    def _shaped_reward(self, reward: float, action: int, env: Any) -> float:
        """
        Adds a shaping bonus proportional to the test's failure-rate/cost ratio,
        guiding the agent toward the greedy-failure-rate oracle during early training.
        The bonus decays as epsilon → epsilon_min (exploration diminishes).
        """
        fail_rate = float(self.test_history.get(action, 0.05))
        cost = max(1e-4, float(env.build.costs[action]))
        shaping_bonus = self.shaping_coef * (fail_rate / cost)
        # Anneal shaping bonus with epsilon so it fades as agent matures
        epsilon_frac = max(0.0, (self.epsilon - self.epsilon_min) / max(1e-9, 1.0 - self.epsilon_min))
        return reward + epsilon_frac * shaping_bonus

    def update(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        next_legal_actions: List[int],
        terminated: bool,
        env: Any = None,
    ) -> float:
        """
        Stores transition in replay buffer and performs batch gradient updates.
        Returns mean absolute TD error of the batch.
        """
        if env is None:
            return 0.0

        shaped_r = self._shaped_reward(reward, action, env)

        # Store features in replay buffer (not raw states)
        feat = extract_test_features(action, env, self.test_history)
        # For next state we store a snapshot of next_legal_action features
        next_feats = []
        if not terminated and next_legal_actions and env is not None:
            next_feats = [
                extract_test_features(a, env, self.test_history)
                for a in next_legal_actions
            ]
        self.replay.push((feat, shaped_r, next_feats, terminated))

        mean_td = 0.0
        if len(self.replay) >= self.batch_size:
            mean_td = self._update_batch()

        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self.w_target = self.w.copy()

        return mean_td

    def _update_batch(self) -> float:
        """Performs one mini-batch gradient descent update on the replay buffer."""
        transitions, indices, importance_weights = self.replay.sample(
            self.batch_size, beta=self.per_beta
        )
        self.per_beta = min(self.per_beta_end, self.per_beta + self.per_beta_step)

        td_errors = []
        grad_accum = np.zeros(NUM_FEATURES)

        for (feat, reward, next_feats, terminated), iw in zip(transitions, importance_weights):
            q_current = self._q_value(feat)
            if terminated or not next_feats:
                target = reward
            else:
                # Double Q: use w to choose best action, w_target to evaluate it
                best_idx = int(np.argmax([self._q_value(nf) for nf in next_feats]))
                q_next = self._q_value(next_feats[best_idx], self.w_target)
                target = reward + self.gamma * q_next

            td_err = target - q_current
            td_errors.append(abs(td_err))
            grad_accum += iw * td_err * feat

        # Semi-gradient update
        self.w += (self.alpha / self.batch_size) * grad_accum

        # Clip weights to prevent divergence
        self.w = np.clip(self.w, -50.0, 50.0)

        self.replay.update_priorities(indices, td_errors)
        return float(np.mean(td_errors))

    def decay_epsilon(self) -> float:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
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
            self.w = data["w"]
            self.w_target = data.get("w_target", self.w.copy())
            self.epsilon = data.get("epsilon", self.epsilon_min)
            self.gamma = data.get("gamma", self.gamma)
            self.alpha = data.get("alpha", self.alpha)
        else:
            # Trying to load tabular model into linear — reinitialize silently
            pass


# ─────────────────────────────────────────────────────────
# Legacy Tabular Agent (kept for ablation study)
# ─────────────────────────────────────────────────────────

class TabularQLearningAgent:
    """
    Original tabular Q-learning agent.
    Kept for ablation comparison against the LinearQLearningAgent.

    Uses a 4D Q-table: [budget_bin, coverage_bin, history_bin, action]
    Combined with a greedy-style priority score during action selection.
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
        self.num_actions = num_actions
        self.budget_bins = budget_bins
        self.coverage_bins = coverage_bins
        self.history_bins = history_bins

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.test_history = test_history or {}
        self.w_fail = float(w_fail)
        self.w_cov = float(w_cov)

        self.rng = np.random.default_rng(seed)

        # 4D Q-table: [budget_bin, coverage_bin, history_bin, action]
        self.q_table = np.zeros(
            (budget_bins, coverage_bins, history_bins, num_actions), dtype=np.float64
        )

    def set_test_history(self, test_history: Dict[int, float]):
        self.test_history = test_history

    def get_action_value(self, state: Tuple[int, int, int], action: int, env: Any = None) -> float:
        b, c, h = state
        act_idx = min(action, self.num_actions - 1)
        base_q = self.q_table[b, c, h, act_idx]

        if env is not None and env.build is not None:
            build = env.build
            cost = max(1e-4, build.costs[action])
            h_rate = self.test_history.get(action, 0.05)
            new_feats = len(build.coverage_map.get(action, set()) - env.covered_features)

            fail_eff = (h_rate + 1e-4) / cost
            cov_eff = new_feats / cost
            order_prio = 1.0 - (action / max(1, build.num_tests))

            return float(base_q + self.w_fail * fail_eff + self.w_cov * cov_eff + 0.5 * order_prio)

        return float(base_q)

    def select_action(
        self,
        state: Tuple[int, int, int],
        legal_actions: List[int],
        explore: bool = True,
        env: Any = None,
    ) -> int:
        if not legal_actions:
            raise ValueError("No legal actions available to select from.")

        if explore and (self.rng.random() < self.epsilon):
            return int(self.rng.choice(legal_actions))

        b, c, h = state
        q_vals = [self.get_action_value(state, a, env=env) for a in legal_actions]
        max_q = max(q_vals)

        best_actions = [a for a, q in zip(legal_actions, q_vals) if np.isclose(q, max_q, atol=1e-8)]
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
        b, c, h = state
        act_idx = min(action, self.num_actions - 1)
        current_q = self.q_table[b, c, h, act_idx]

        if terminated or not next_legal_actions:
            target = reward
        else:
            nb, nc, nh = next_state
            max_next_q = max(self.q_table[nb, nc, nh, min(a, self.num_actions - 1)] for a in next_legal_actions)
            target = reward + self.gamma * max_next_q

        td_error = target - current_q
        self.q_table[b, c, h, act_idx] += self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self) -> float:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
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
            self.epsilon = data.get("epsilon", self.epsilon_min)
