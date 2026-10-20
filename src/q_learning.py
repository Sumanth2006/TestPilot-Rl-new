"""
Tabular Q-Learning Agent with Legal Action Masking for TestPilot-RL.

Implements:
- Tabular Q-value storage across discrete states (budget_bin, coverage_bin, history_bin)
- Epsilon-greedy action selection strictly restricted to legal actions
- Standard Bellman update equation:
    Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
- Correct terminal handling (future value = 0 when terminated or no legal next actions)
- Model saving, loading, and inspection
"""

import os
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class TabularQLearningAgent:
    """
    Q-learning agent with legal action masking and state-action value table.
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

        self.rng = np.random.default_rng(seed)

        # 4D Q-table: [budget_bin, coverage_bin, history_bin, action]
        self.q_table = np.zeros(
            (budget_bins, coverage_bins, history_bins, num_actions), dtype=np.float64
        )

    def select_action(
        self,
        state: Tuple[int, int, int],
        legal_actions: List[int],
        explore: bool = True,
    ) -> int:
        """
        Selects an action using epsilon-greedy policy over legal actions only.
        """
        if not legal_actions:
            raise ValueError("No legal actions available to select from.")

        if explore and (self.rng.random() < self.epsilon):
            # Uniform exploration strictly over legal actions
            return int(self.rng.choice(legal_actions))

        # Exploitation: Select action with highest Q-value among legal actions
        b, c, h = state
        q_vals = self.q_table[b, c, h]

        # Extract Q-values only for legal actions
        legal_q = [q_vals[a] for a in legal_actions]
        max_q = max(legal_q)

        # Break ties randomly among best actions
        best_actions = [a for a in legal_actions if np.isclose(q_vals[a], max_q, atol=1e-8)]
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: Tuple[int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int],
        next_legal_actions: List[int],
        terminated: bool,
    ) -> float:
        """
        Performs the Bellman Q-value update.
        Returns the temporal difference (TD) error.
        """
        b, c, h = state
        current_q = self.q_table[b, c, h, action]

        # Terminal state handling: Future value is zero if terminated or no next legal action
        if terminated or not next_legal_actions:
            target = reward
        else:
            nb, nc, nh = next_state
            next_q_vals = self.q_table[nb, nc, nh]
            # Max future Q strictly over valid legal actions in next state
            max_next_q = max(next_q_vals[a] for a in next_legal_actions)
            target = reward + self.gamma * max_next_q

        td_error = target - current_q
        self.q_table[b, c, h, action] += self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self) -> float:
        """Decays exploration parameter epsilon toward epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, filepath: str):
        """Saves Q-table and hyperparameters to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
            },
            allow_pickle=True,
        )

    def load(self, filepath: str):
        """Loads Q-table from disk."""
        data = np.load(filepath, allow_pickle=True).item()
        self.q_table = data["q_table"]
        self.num_actions = data["num_actions"]
        self.budget_bins = data["budget_bins"]
        self.coverage_bins = data["coverage_bins"]
        self.history_bins = data["history_bins"]
        self.alpha = data.get("alpha", self.alpha)
        self.gamma = data.get("gamma", self.gamma)
        self.epsilon = data.get("epsilon", self.epsilon_min)
