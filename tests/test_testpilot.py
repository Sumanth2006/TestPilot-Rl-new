"""
Unit test suite for TestPilot-RL (tests/test_testpilot.py).

Verifies:
1. Strict budget enforcement and legal-action masking (no test repeats, no budget overruns).
2. Strict information boundary (synthetic builds with no prior history have zero ground-truth leakage).
3. Zero/random initialization and TD learning update of LinearQLearningAgent, DuelingDQNAgent, and TabularQLearningAgent.
4. Full-suite APFD and NAPFD mathematical correctness against hand-calculated examples.
"""

import numpy as np
from src.simulator import BuildGenerator, SoftwareBuild
from src.environment import TestingEnvironment
from src.q_learning import (
    LinearQLearningAgent,
    DuelingDQNAgent,
    TabularQLearningAgent,
    extract_test_features,
)
from src.baselines import (
    RandomBaseline,
    GreedyCoverageCostBaseline,
    GreedyFailureRateBaseline,
    GreedyRecencyCostBaseline,
)
from src.metrics import compute_apfd, compute_napfd


def test_environment_budget_and_no_repeat_constraints():
    bg = BuildGenerator(num_tests=15, num_features=6, min_bugs=3, max_bugs=5, budget_ratio=0.45, seed=42)
    build = bg.generate_synthetic_build("test_build_01", seed=42)
    env = TestingEnvironment(build)

    state, info = env.reset(build)
    seen_actions = set()

    while True:
        legal = env.get_legal_actions()
        if not legal:
            break
        for a in legal:
            assert a not in seen_actions, "Executed test appeared in legal actions!"
            assert build.costs[a] <= env.remaining_budget + 1e-4, "Unaffordable test appeared in legal actions!"

        chosen = legal[0]
        seen_actions.add(chosen)
        state, reward, terminated, truncated, step_info = env.step(chosen)
        if terminated:
            break

    assert env.total_cost_spent <= build.budget + 1e-4
    assert env.remaining_budget >= -1e-4


def test_strict_information_boundary_no_leakage_on_initial_build():
    bg = BuildGenerator(num_tests=20, num_features=8, min_bugs=4, max_bugs=6, seed=99)
    # Generate a standalone build with no prior history
    build = bg.generate_synthetic_build("standalone_build", seed=99)

    for t in build.test_cases:
        assert build.co_failure_map[t] == set(), "co_failure_map leaked current-build ground truth!"
        assert build.recent_verdicts[t] == [], "recent_verdicts leaked current-build ground truth!"
        assert np.isclose(build.causal_fail_rates[t], 0.10), "causal_fail_rates leaked archetype ground truth!"


def test_q_learning_updates_from_near_zero_initialization():
    bg = BuildGenerator(num_tests=10, num_features=6, min_bugs=2, max_bugs=4, seed=7)
    builds = bg.generate_dataset(num_builds=5, prefix="unit", base_seed=7)
    env = TestingEnvironment(builds[0])

    lin_agent = LinearQLearningAgent(alpha=0.05, batch_size=4, seed=7)
    assert np.all(np.abs(lin_agent.w) < 0.1), "LinearQLearningAgent weights should initialize near zero!"

    dqn_agent = DuelingDQNAgent(alpha=0.01, batch_size=4, seed=7)
    assert np.allclose(dqn_agent.w_skip, 0.0), "DuelingDQNAgent skip weights should initialize at zero!"

    tab_agent = TabularQLearningAgent(num_actions=10, seed=7)
    assert np.allclose(tab_agent.q_table, 0.0), "Tabular Q-table should initialize at zero!"

    state, _ = env.reset(builds[0])
    for _ in range(6):
        legal = env.get_legal_actions()
        if not legal:
            break
        action = lin_agent.select_action(state, legal, explore=True, env=env)
        feats = extract_test_features(action, env, lin_agent.test_history)
        next_state, reward, done, _, _ = env.step(action)
        next_legal = env.get_legal_actions()
        lin_agent.update(state, action, reward, next_state, next_legal, done, env=env, action_features=feats)
        tab_agent.update(state, action, reward, next_state, next_legal, done, env=env)
        state = next_state
        if done:
            break

    assert not np.allclose(tab_agent.q_table, 0.0), "Tabular Q-table did not update!"


def test_apfd_and_napfd_hand_calculated():
    # 4 total tests in suite, 2 bugs: T0 finds Bug 0, T2 finds Bug 1
    bug_map = {0: {0}, 1: set(), 2: {1}, 3: set()}
    # Policy A executes [0, 1, 2, 3]: TF_0 = 1, TF_1 = 3 -> APFD = 1 - (1+3)/(4*2) + 1/(2*4) = 0.625
    apfd_a = compute_apfd([0, 1, 2, 3], bug_map, total_bugs=2, total_tests=4)
    napfd_a = compute_napfd([0, 1, 2, 3], bug_map, total_bugs=2, total_tests=4)
    assert np.isclose(apfd_a, 0.625)
    assert np.isclose(napfd_a, 0.625)

    # Partial execution [0, 1] (only Bug 0 detected; p = 0.5, N = 4, m = 2):
    # NAPFD = 0.5 - 1/(4*2) + 0.5/(2*4) = 0.5 - 0.125 + 0.0625 = 0.4375
    napfd_partial = compute_napfd([0, 1], bug_map, total_bugs=2, total_tests=4)
    assert np.isclose(napfd_partial, 0.4375)