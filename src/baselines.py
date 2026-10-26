"""
Baseline policies for TestPilot-RL.

Implements fair, budget-constrained baselines:
1. RandomBaseline: Uniform random choice among currently legal actions.
2. GreedyCoverageCostBaseline: Greedy heuristic prioritizing maximum observable feature coverage per cost unit.
3. GreedyFailureRateBaseline: Prioritizes static training-set historical failure rate per cost unit.
4. GreedyRecencyCostBaseline: Symmetric recency-aware + online-updating heuristic prioritizing
   combined recent/historical failure rate per cost unit (controls for recency access).
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
    Prioritizes tests based on static training-set historical failure frequency
    divided by cost. Uses canonical test identity lookup via build.lookup_history()
    to prevent cycle-local index mismatches.
    """

    def __init__(self, test_history: Optional[Dict[Any, float]] = None):
        self.test_history = dict(test_history) if test_history else {}
        self.prior_rate = (
            float(np.mean(list(self.test_history.values())))
            if self.test_history
            else 0.10
        )

    def set_history(self, test_history: Dict[Any, float]):
        self.test_history = dict(test_history) if test_history else {}
        self.prior_rate = (
            float(np.mean(list(self.test_history.values())))
            if self.test_history
            else 0.10
        )

    def select_action(self, env: TestingEnvironment) -> int:
        legal = env.get_legal_actions()
        if not legal:
            raise ValueError("No legal actions available for GreedyFailureRate baseline.")

        build = env.build
        best_action = legal[0]
        best_score = -1.0

        for action in legal:
            if hasattr(build, "lookup_history"):
                hist_rate = build.lookup_history(action, self.test_history, default=self.prior_rate)
            else:
                hist_rate = float(self.test_history.get(action, self.prior_rate))
            cost = max(1e-5, build.costs[action])
            score = (hist_rate + 1e-4) / cost

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    @property
    def name(self) -> str:
        return "Greedy (Failure/Cost)"


class GreedyRecencyCostBaseline(BasePolicy):
    """
    Symmetric recency-aware heuristic baseline that accesses the exact same
    causal `recent_verdicts` and online history updates as the RL agents,
    ranking tests by combined recent + long-term failure rate divided by cost
    (without RL's sequential coverage-diversity and state-value learning).
    """

    def __init__(self, test_history: Optional[Dict[Any, float]] = None):
        self.test_history = dict(test_history) if test_history else {}
        self.prior_rate = (
            float(np.mean(list(self.test_history.values())))
            if self.test_history
            else 0.10
        )

    def set_test_history(self, test_history: Dict[Any, float]):
        self.test_history = dict(test_history) if test_history else {}

    def select_action(self, env: TestingEnvironment) -> int:
        legal = env.get_legal_actions()
        if not legal:
            raise ValueError("No legal actions available for GreedyRecencyCost baseline.")

        build = env.build
        best_action = legal[0]
        best_score = -1.0

        for action in legal:
            if hasattr(build, "lookup_history"):
                long_rate = build.lookup_history(action, self.test_history, default=self.prior_rate)
            else:
                long_rate = float(self.test_history.get(action, self.prior_rate))

            recent_list = getattr(build, "recent_verdicts", {}).get(action, [])
            if recent_list:
                n_r = min(5, len(recent_list))
                w = np.array([0.50, 0.25, 0.12, 0.08, 0.05][:n_r], dtype=np.float64)
                w /= w.sum()
                rec_rate = float(np.dot(w, recent_list[:n_r]))
            else:
                rec_rate = long_rate

            eff_rate = 0.35 * long_rate + 0.65 * rec_rate
            cost = max(1e-5, build.costs[action])
            score = (eff_rate + 1e-4) / cost

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def adapt_online(self, build: Any, executed_actions: List[int]) -> None:
        """Online causal history update identical to RL agents."""
        for action in executed_actions:
            failed = 1.0 if len(build.bug_detection_map.get(action, set())) > 0 else 0.0
            keys = set()
            if hasattr(build, "test_names") and action in build.test_names:
                keys.add(build.test_names[action])
            if hasattr(build, "global_test_ids") and action in build.global_test_ids:
                keys.add(build.global_test_ids[action])
            if not keys:
                keys.add(action)
            for k in keys:
                old_rate = float(self.test_history.get(k, self.prior_rate))
                self.test_history[k] = 0.80 * old_rate + 0.20 * failed

    @property
    def name(self) -> str:
        return "Greedy (Recency/Cost)"