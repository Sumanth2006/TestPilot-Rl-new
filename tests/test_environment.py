"""
Unit and integration tests for TestingEnvironment and SoftwareBuild simulator.
Verifies budget constraints, legal action masking, and information boundary.
"""

import pytest
import numpy as np
from src.simulator import BuildGenerator, SoftwareBuild
from src.environment import TestingEnvironment


@pytest.fixture
def sample_build():
    bg = BuildGenerator(num_tests=10, num_features=5, min_bugs=3, max_bugs=3, budget_ratio=0.5, seed=123)
    return bg.generate_synthetic_build("test_build_01")


def test_environment_initialization(sample_build):
    env = TestingEnvironment(sample_build)
    state, info = env.reset()

    assert isinstance(state, tuple)
    assert len(state) == 3
    assert info["total_hidden_bugs"] == 3
    assert info["unique_bugs_detected"] == 0
    assert info["coverage_rate"] == 0.0
    assert info["budget_consumed"] == 0.0
    assert info["remaining_budget"] == sample_build.budget


def test_budget_never_negative(sample_build):
    env = TestingEnvironment(sample_build)
    state, info = env.reset()

    step = 0
    while True:
        legals = env.get_legal_actions()
        if not legals:
            break
        # Pick any legal action
        action = legals[0]
        state, reward, terminated, truncated, info = env.step(action)
        assert env.remaining_budget >= 0.0, f"Remaining budget became negative: {env.remaining_budget}"
        step += 1
        if terminated:
            break

    assert step > 0
    assert env.remaining_budget >= 0.0


def test_no_test_executed_twice(sample_build):
    env = TestingEnvironment(sample_build)
    env.reset()

    legals = env.get_legal_actions()
    first_action = legals[0]
    env.step(first_action)

    assert first_action not in env.get_legal_actions(), "Executed test was still in legal actions!"
    assert first_action in env.executed_tests


def test_illegal_action_raises_exception(sample_build):
    env = TestingEnvironment(sample_build)
    env.reset()

    legals = env.get_legal_actions()
    action = legals[0]
    env.step(action)

    # Attempting to execute the same test again must raise ValueError
    with pytest.raises(ValueError, match="is ILLEGAL"):
        env.step(action)


def test_information_boundary_no_leakage(sample_build):
    """
    Ensures state and observation vectors NEVER contain hidden bug IDs
    or test-to-bug mappings.
    """
    env = TestingEnvironment(sample_build)
    state, info = env.reset()
    obs = env.get_continuous_observation()

    # Discrete state has 3 binned integers
    assert len(state) == 3
    for s in state:
        assert isinstance(s, (int, np.integer))

    # Observation vector contains only observable scalars
    assert len(obs) == 5
    assert not any(np.isnan(obs))

    # Check that info at reset doesn't expose which tests detect bugs
    assert "bug_detection_map" not in info
    assert "hidden_bugs" not in info


def test_reward_calculation():
    # Build with known cost, bug, and coverage
    b = SoftwareBuild(
        build_id="fixed",
        test_cases=[0, 1],
        costs=[2.0, 3.0],
        coverage_map={0: {0, 1}, 1: {2}},
        hidden_bugs={0, 1},
        bug_detection_map={0: {0}, 1: {1}},
        budget=5.0,
        total_features=4,
    )
    env = TestingEnvironment(b, w_b=2.0, w_c=1.0, w_e=0.5)
    env.reset()

    # Action 0: 1 new bug (b_term=1), 2 new feats / 4 = 0.5 (c_term=0.5), cost 2/5 = 0.4 (e_term=0.4)
    # Expected reward: 2.0*1 + 1.0*0.5 - 0.5*0.4 = 2.0 + 0.5 - 0.2 = 2.3
    state, reward, term, trunc, info = env.step(0)
    expected_r = (2.0 * 1.0) + (1.0 * (2 / 4)) - (0.5 * (2.0 / 5.0))
    assert pytest.approx(reward, 0.001) == expected_r
