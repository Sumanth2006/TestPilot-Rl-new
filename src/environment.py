"""
RL Testing Environment for TestPilot-RL.
Follows Gym-like API (reset, step, get_legal_actions).

Strict Information Boundary:
- The agent NEVER observes the hidden test-to-bug mapping or future verdicts.
- The simulator only reveals outcomes AFTER a legal test action is executed.
- Action masking enforces:
  1. No test is executed more than once per episode.
  2. No test can be selected if its cost exceeds the remaining budget.
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from src.simulator import SoftwareBuild


class TestingEnvironment:
    """
    Simulated software testing environment with budget constraints,
    code coverage tracking, and defect discovery.
    """
    __test__ = False

    def __init__(
        self,
        build: Optional[SoftwareBuild] = None,
        w_b: float = 3.0,
        w_c: float = 0.8,
        w_e: float = 0.25,
        budget_bins: int = 5,
        coverage_bins: int = 5,
        history_bins: int = 3,
    ):
        self.build = build
        self.w_b = w_b
        self.w_c = w_c
        self.w_e = w_e
        self.budget_bins = budget_bins
        self.coverage_bins = coverage_bins
        self.history_bins = history_bins

        # Episode state variables
        self.executed_tests: Set[int] = set()
        self.failed_executed_tests: Set[int] = set()
        self.passed_executed_tests: Set[int] = set()
        self.discovered_bugs: Set[int] = set()
        self.covered_features: Set[int] = set()
        self.remaining_budget: float = 0.0
        self.total_budget: float = 0.0
        self.total_cost_spent: float = 0.0
        self.recent_outcomes: List[int] = []  # 1 for bug found, 0 otherwise
        self.step_count: int = 0
        self._cached_legal_actions: Optional[List[int]] = None

        if build is not None:
            self.reset(build)

    def set_build(self, build: SoftwareBuild):
        """Set a new build scenario for the environment."""
        self.build = build
        self._cached_legal_actions = None

    def reset(self, build: Optional[SoftwareBuild] = None) -> Tuple[Tuple[int, int, int], Dict[str, Any]]:
        """
        Resets the environment for a new testing episode.
        Returns:
            state: Compact discrete state tuple (budget_bin, coverage_bin, history_bin)
            info: Initial diagnostic information
        """
        if build is not None:
            self.build = build

        if self.build is None:
            raise ValueError("No SoftwareBuild provided to environment.")

        self.executed_tests = set()
        self.failed_executed_tests = set()
        self.passed_executed_tests = set()
        self.discovered_bugs = set()
        self.covered_features = set()
        self.total_budget = float(self.build.budget)
        self.remaining_budget = float(self.build.budget)
        self.total_cost_spent = 0.0
        self.recent_outcomes = []
        self.step_count = 0
        self._cached_legal_actions = None

        state = self.get_discrete_state()
        info = self._get_info()
        return state, info

    def get_legal_actions(self) -> List[int]:
        """
        Returns list of test case indices that:
        1. Have NOT been executed yet in this episode.
        2. Have cost <= remaining budget.
        """
        if self.build is None:
            return []

        if self._cached_legal_actions is not None:
            return self._cached_legal_actions

        rem = self.remaining_budget + 1e-4
        costs = self.build.costs
        executed = self.executed_tests
        legal = [t for t in self.build.test_cases if t not in executed and costs[t] <= rem]
        self._cached_legal_actions = legal
        return legal

    def get_legal_action_mask(self) -> np.ndarray:
        """
        Returns boolean array of shape (num_tests,) where True indicates legal action.
        """
        if self.build is None:
            return np.zeros(0, dtype=bool)

        mask = np.zeros(self.build.num_tests, dtype=bool)
        for a in self.get_legal_actions():
            mask[a] = True
        return mask

    def get_discrete_state(self) -> Tuple[int, int, int]:
        """
        Produces a compact, meaningful discrete state representation:
        - budget_bin: [0, budget_bins - 1] based on remaining budget fraction
        - coverage_bin: [0, coverage_bins - 1] based on features covered fraction
        - history_bin: [0, history_bins - 1] based on recent bug detection momentum
        """
        if self.total_budget <= 0:
            b_ratio = 0.0
        else:
            b_ratio = max(0.0, min(1.0, self.remaining_budget / self.total_budget))

        budget_bin = min(self.budget_bins - 1, int(b_ratio * self.budget_bins))

        total_feats = max(1, self.build.total_features) if self.build else 1
        c_ratio = max(0.0, min(1.0, len(self.covered_features) / total_feats))
        coverage_bin = min(self.coverage_bins - 1, int(c_ratio * self.coverage_bins))

        # History bin based on last 3 executions
        if not self.recent_outcomes:
            history_bin = 0
        else:
            recent_bugs = sum(self.recent_outcomes[-3:])
            if recent_bugs == 0:
                history_bin = 0
            elif recent_bugs == 1:
                history_bin = 1
            else:
                history_bin = min(self.history_bins - 1, 2)

        return (budget_bin, coverage_bin, history_bin)

    def get_continuous_observation(self) -> np.ndarray:
        """
        Continuous observation vector (without leaking hidden bugs):
        [remaining_budget_ratio, coverage_ratio, bugs_found_normalized, test_executed_ratio, last_test_found_bug]
        """
        total_feats = max(1, self.build.total_features) if self.build else 1
        num_tests = max(1, self.build.num_tests) if self.build else 1
        b_ratio = max(0.0, self.remaining_budget / max(1e-5, self.total_budget))
        c_ratio = len(self.covered_features) / total_feats
        bugs_norm = len(self.discovered_bugs) / max(1.0, num_tests * 0.3)
        exec_ratio = len(self.executed_tests) / num_tests
        last_found = float(self.recent_outcomes[-1]) if self.recent_outcomes else 0.0

        return np.array([b_ratio, c_ratio, bugs_norm, exec_ratio, last_found], dtype=np.float32)

    def step(
        self, action: int
    ) -> Tuple[Tuple[int, int, int], float, bool, bool, Dict[str, Any]]:
        """
        Executes a test case action.
        Returns:
            next_state: compact discrete tuple
            reward: scalar reward according to cost-aware & early-detection objective
            terminated: True if episode complete (budget depleted / no legal actions)
            truncated: False (standard termination)
            info: comprehensive step and episode metrics
        """
        if self.build is None:
            raise ValueError("Environment not initialized with a build.")

        legal_actions = self.get_legal_actions()
        if action not in legal_actions:
            raise ValueError(
                f"Action {action} is ILLEGAL. Legal actions: {legal_actions}. "
                f"Remaining budget: {self.remaining_budget:.2f}, Cost: {self.build.costs[action]:.2f}, "
                f"Already executed: {action in self.executed_tests}"
            )

        cost = float(self.build.costs[action])
        pre_step_budget_ratio = self.remaining_budget / max(1.0, self.total_budget)

        self.remaining_budget = max(0.0, round(self.remaining_budget - cost, 4))
        self.total_cost_spent += cost
        self.executed_tests.add(action)
        self.step_count += 1
        self._cached_legal_actions = None

        # Query simulator ground truth for newly discovered bugs
        revealed_bugs = self.build.bug_detection_map.get(action, set())
        new_bugs = revealed_bugs - self.discovered_bugs
        self.discovered_bugs.update(new_bugs)

        # Track whether this executed test failed or passed (observable after execution)
        if len(revealed_bugs) > 0:
            self.failed_executed_tests.add(action)
        else:
            self.passed_executed_tests.add(action)

        # Query simulator ground truth for newly covered features
        test_feats = self.build.coverage_map.get(action, set())
        new_features = test_feats - self.covered_features
        self.covered_features.update(new_features)

        # Record recent outcome
        num_new_bugs = len(new_bugs)
        self.recent_outcomes.append(1 if num_new_bugs > 0 else 0)

        # Cost-aware & early-detection (APFD-aligned) reward formulation:
        # Rewards discovering unique new bugs early in the budget and at low execution cost,
        # while encouraging marginal feature coverage diversity.
        total_feats = max(1, self.build.total_features)
        early_bonus = 1.0 + 0.5 * pre_step_budget_ratio
        cost_efficiency_factor = 1.0 / max(0.5, np.sqrt(cost))
        b_term = float(num_new_bugs) * early_bonus * (0.65 + 0.35 * cost_efficiency_factor)
        c_term = (float(len(new_features)) / total_feats) * cost_efficiency_factor
        e_term = cost / max(1.0, self.total_budget)

        reward = (self.w_b * b_term) + (self.w_c * c_term) - (self.w_e * e_term)

        # Check termination: no legal actions left or budget reached 0
        remaining_legal = self.get_legal_actions()
        terminated = (len(remaining_legal) == 0) or (self.remaining_budget <= 0.0)
        truncated = False

        next_state = self.get_discrete_state()
        info = self._get_info()
        info.update({
            "action_executed": action,
            "action_cost": cost,
            "new_bugs_found": num_new_bugs,
            "new_features_covered": len(new_features),
            "step_reward": reward,
        })

        return next_state, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """Compiles current episode performance metrics."""
        total_bugs = len(self.build.hidden_bugs) if self.build else 0
        total_feats = self.build.total_features if self.build else 1

        bdr = (len(self.discovered_bugs) / total_bugs) if total_bugs > 0 else 1.0
        cov = (len(self.covered_features) / total_feats) if total_feats > 0 else 0.0
        efficiency = (len(self.discovered_bugs) / self.total_cost_spent) if self.total_cost_spent > 0 else 0.0

        return {
            "build_id": self.build.build_id if self.build else "",
            "unique_bugs_detected": len(self.discovered_bugs),
            "total_hidden_bugs": total_bugs,
            "bug_detection_rate": bdr,
            "features_covered": len(self.covered_features),
            "total_features": total_feats,
            "coverage_rate": cov,
            "budget_consumed": round(self.total_cost_spent, 2),
            "remaining_budget": round(self.remaining_budget, 2),
            "total_budget": self.total_budget,
            "budget_consumed_ratio": (self.total_cost_spent / self.total_budget) if self.total_budget > 0 else 0.0,
            "bugs_per_cost_unit": efficiency,
            "tests_executed_count": len(self.executed_tests),
            "total_tests_available": self.build.num_tests if self.build else 0,
        }