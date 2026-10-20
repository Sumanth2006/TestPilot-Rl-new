"""
Baseline policies for TestPilot-RL.

Implements fair, budget-constrained baselines:
1. RandomSelection: Uniform random choice among currently legal actions.
2. GreedyCoverageCost: Greedy heuristic prioritizing maximum observable feature coverage per cost unit.
3. GreedyFailureRate: Prioritizes historical failure rate per cost unit (standard RETECS benchmark heuristic).
4. GreedyHybrid: Balanced observable score prioritizing both coverage and historical failure rate.
"""

import numpy as np
from typing import List, Dict, Set, Optional, Any
from src.environment import TestingEnvironment


class BasePolicy:
    """Abstract base class for all testing policies."""

    def select_action(self, env: TestingEnvironment) -> int:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class RandomBaseline(BasePolicy):
    """
    Selects uniformly at random among currently legal actions.
    Follows identical budget and masking constraints as RL agent.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def select_action(self, env: TestingEnvironment) -> int:
        legal = env.get_legal_actions()
        if not legal:
            raise ValueError("No legal actions available for Random baseline.")
        return int(self.rng.choice(legal))

    @property
    def name(self) -> str:
        return "Random"


class GreedyCoverageCostBaseline(BasePolicy):
    """
    Greedy heuristic selecting the legal test that maximizes:
    observable new feature coverage / execution cost.

    Does NOT use hidden bug mappings (strictly adheres to information boundary).
    """

    def select_action(self, env: TestingEnvironment) -> int:
        legal = env.get_legal_actions()
        if not legal:
            raise ValueError("No legal actions available for GreedyCoverageCost baseline.")

        build = env.build
        best_action = legal[0]
        best_score = -1.0

        for action in legal:
            test_feats = build.coverage_map.get(action, set())
            # Observable potential new features (based on features not yet covered)
            new_feats = len(test_feats - env.covered_features)
            cost = max(1e-5, build.costs[action])
            score = new_feats / cost

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    @property
    def name(self) -> str:
        return "Greedy (Coverage/Cost)"


class GreedyFailureRateBaseline(BasePolicy):
    """
    Prioritizes tests based on historical failure frequency divided by cost.
    Standard RETECS baseline heuristic.
    """

    def __init__(self, test_history: Optional[Dict[int, float]] = None):
        # test_history: mapping test_id -> historical failure rate [0, 1]
        self.test_history = test_history or {}

    def set_history(self, test_history: Dict[int, float]):
        self.test_history = test_history

    def select_action(self, env: TestingEnvironment) -> int:
        legal = env.get_legal_actions()
        if not legal:
            raise ValueError("No legal actions available for GreedyFailureRate baseline.")

        build = env.build
        best_action = legal[0]
        best_score = -1.0

        for action in legal:
            hist_rate = self.test_history.get(action, 0.05)
            cost = max(1e-5, build.costs[action])
            score = (hist_rate + 1e-4) / cost

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    @property
    def name(self) -> str:
        return "Greedy (Failure/Cost)"
