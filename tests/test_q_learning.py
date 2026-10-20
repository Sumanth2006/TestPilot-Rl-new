"""
Unit tests for TabularQLearningAgent.
Verifies Q-table initialization, epsilon-greedy action masking, Bellman update,
terminal handling, and serialization.
"""

import os
import pytest
import numpy as np
from src.q_learning import TabularQLearningAgent


def test_q_table_shape():
    agent = TabularQLearningAgent(num_actions=10, budget_bins=4, coverage_bins=4, history_bins=3)
    assert agent.q_table.shape == (4, 4, 3, 10)
    assert np.all(agent.q_table == 0.0)


def test_action_selection_strictly_legal():
    agent = TabularQLearningAgent(num_actions=10, epsilon_start=0.0)  # Greedy mode
    # Artificially set high Q-value on an illegal action
    state = (2, 2, 0)
    agent.q_table[2, 2, 0, 7] = 999.0  # action 7 is highest
    agent.q_table[2, 2, 0, 3] = 10.0   # action 3 is legal and second best

    legal_actions = [1, 2, 3, 4]  # 7 is ILLEGAL
    chosen = agent.select_action(state, legal_actions, explore=False)
    assert chosen == 3, f"Agent chose {chosen} instead of best legal action 3!"
    assert chosen in legal_actions


def test_bellman_update_non_terminal():
    agent = TabularQLearningAgent(alpha=0.5, gamma=0.8)
    state = (1, 1, 0)
    next_state = (2, 2, 0)
    action = 2

    # Set Q-values in next state
    agent.q_table[2, 2, 0, 4] = 10.0
    agent.q_table[2, 2, 0, 5] = 20.0  # Max next legal Q = 20.0
    next_legal = [4, 5]

    # Target = reward (5.0) + gamma (0.8) * max_next_q (20.0) = 5.0 + 16.0 = 21.0
    # Current Q = 0.0 -> Updated Q = 0.0 + 0.5 * (21.0 - 0.0) = 10.5
    td_err = agent.update(state, action, reward=5.0, next_state=next_state, next_legal_actions=next_legal, terminated=False)

    assert pytest.approx(agent.q_table[1, 1, 0, 2], 0.01) == 10.5
    assert pytest.approx(td_err, 0.01) == 21.0


def test_bellman_update_terminal_sets_future_zero():
    agent = TabularQLearningAgent(alpha=0.5, gamma=0.9)
    state = (1, 0, 0)
    next_state = (0, 0, 0)
    action = 1

    # In next state, Q has values, but terminated=True means future term MUST be zero
    agent.q_table[0, 0, 0, 3] = 100.0
    td_err = agent.update(state, action, reward=4.0, next_state=next_state, next_legal_actions=[3], terminated=True)

    # Updated Q = 0 + 0.5 * (4.0 - 0) = 2.0
    assert pytest.approx(agent.q_table[1, 0, 0, 1], 0.01) == 2.0


def test_epsilon_decay():
    agent = TabularQLearningAgent(epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.5)
    agent.decay_epsilon()
    assert pytest.approx(agent.epsilon, 0.01) == 0.5
    agent.decay_epsilon()
    assert pytest.approx(agent.epsilon, 0.01) == 0.25
    agent.decay_epsilon()
    assert pytest.approx(agent.epsilon, 0.01) == 0.125
    agent.decay_epsilon()
    assert pytest.approx(agent.epsilon, 0.01) == 0.1  # Capped at epsilon_min


def test_save_and_load_roundtrip(tmp_path):
    save_file = str(tmp_path / "test_q.npy")
    agent = TabularQLearningAgent(num_actions=5, alpha=0.2)
    agent.q_table[1, 2, 1, 3] = 42.5
    agent.save(save_file)

    loaded_agent = TabularQLearningAgent(num_actions=5)
    loaded_agent.load(save_file)

    assert loaded_agent.q_table.shape == agent.q_table.shape
    assert pytest.approx(loaded_agent.q_table[1, 2, 1, 3]) == 42.5
    assert pytest.approx(loaded_agent.alpha) == 0.2
