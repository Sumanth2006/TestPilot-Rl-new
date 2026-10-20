"""
Simulator module for TestPilot-RL.
Generates synthetic software builds and transforms real CI logs into build scenarios.

Information Boundary Principle:
- Ground truth (actual bugs, test-to-bug mappings, full feature coverage) is maintained
  strictly inside the Simulator/Build data structure.
- The agent only receives observations via the environment after executing a test.
"""

import json
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Optional, Any


@dataclass
class SoftwareBuild:
    """
    Represents a single software build/commit under test.
    Ground truth data:
    - test_cases: List of test IDs [0, 1, ..., N-1]
    - costs: Array of execution costs for each test
    - coverage_map: Dict mapping test_id -> set of covered feature IDs
    - hidden_bugs: Set of all bug IDs present in this build
    - bug_detection_map: Dict mapping test_id -> set of bug IDs it reveals (HIDDEN)
    - budget: Total testing budget available for this build
    - total_features: Total number of features in the software system
    """
    build_id: str
    test_cases: List[int]
    costs: List[float]
    coverage_map: Dict[int, Set[int]]
    hidden_bugs: Set[int]
    bug_detection_map: Dict[int, Set[int]]
    budget: float
    total_features: int

    @property
    def num_tests(self) -> int:
        return len(self.test_cases)

    @property
    def total_cost(self) -> float:
        return float(sum(self.costs))

    @property
    def num_hidden_bugs(self) -> int:
        return len(self.hidden_bugs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert build to serializable dictionary (for test/export)."""
        return {
            "build_id": self.build_id,
            "test_cases": self.test_cases,
            "costs": self.costs,
            "coverage_map": {str(k): list(v) for k, v in self.coverage_map.items()},
            "hidden_bugs": list(self.hidden_bugs),
            "bug_detection_map": {str(k): list(v) for k, v in self.bug_detection_map.items()},
            "budget": self.budget,
            "total_features": self.total_features,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SoftwareBuild":
        """Reconstruct build from serialized dictionary."""
        return cls(
            build_id=data["build_id"],
            test_cases=data["test_cases"],
            costs=data["costs"],
            coverage_map={int(k): set(v) for k, v in data["coverage_map"].items()},
            hidden_bugs=set(data["hidden_bugs"]),
            bug_detection_map={int(k): set(v) for k, v in data["bug_detection_map"].items()},
            budget=float(data["budget"]),
            total_features=int(data["total_features"]),
        )


class BuildGenerator:
    """
    Generates synthetic and real-world CI software builds for evaluation.
    """

    def __init__(
        self,
        num_tests: int = 30,
        num_features: int = 12,
        min_bugs: int = 5,
        max_bugs: int = 10,
        budget_ratio: float = 0.45,
        cost_range: tuple = (1.0, 5.0),
        max_features_per_test: int = 4,
        seed: Optional[int] = None,
    ):
        self.num_tests = num_tests
        self.num_features = num_features
        self.min_bugs = min_bugs
        self.max_bugs = max_bugs
        self.budget_ratio = budget_ratio
        self.cost_range = cost_range
        self.max_features_per_test = max_features_per_test
        self.rng = np.random.default_rng(seed)

    def generate_synthetic_build(self, build_id: str, seed: Optional[int] = None) -> SoftwareBuild:
        """
        Creates a synthetic software build with controlled distributions:
        - Diverse execution costs
        - Overlapping feature coverage
        - Hidden bugs distributed across tests
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng

        test_cases = list(range(self.num_tests))
        costs = [round(float(c), 2) for c in rng.uniform(self.cost_range[0], self.cost_range[1], self.num_tests)]

        # Generate feature coverage: each test exercises 1 to max_features_per_test features
        coverage_map: Dict[int, Set[int]] = {}
        for t in test_cases:
            n_feat = int(rng.integers(1, self.max_features_per_test + 1))
            feats = set(rng.choice(self.num_features, size=n_feat, replace=False).tolist())
            coverage_map[t] = feats

        # Generate hidden bugs for this build
        num_bugs = int(rng.integers(self.min_bugs, self.max_bugs + 1))
        hidden_bugs = set(range(num_bugs))

        # Map tests to bugs they can detect:
        # Some tests detect 0 bugs, some detect 1 or more
        bug_detection_map: Dict[int, Set[int]] = {t: set() for t in test_cases}
        for bug_id in hidden_bugs:
            # 1 to 3 tests are capable of finding this bug
            n_detectors = int(rng.integers(1, 4))
            detectors = rng.choice(test_cases, size=min(n_detectors, self.num_tests), replace=False)
            for d in detectors:
                bug_detection_map[int(d)].add(bug_id)

        total_cost = sum(costs)
        budget = round(total_cost * self.budget_ratio, 2)

        return SoftwareBuild(
            build_id=build_id,
            test_cases=test_cases,
            costs=costs,
            coverage_map=coverage_map,
            hidden_bugs=hidden_bugs,
            bug_detection_map=bug_detection_map,
            budget=budget,
            total_features=self.num_features,
        )

    def generate_dataset(
        self, num_builds: int = 100, prefix: str = "build", base_seed: int = 1000
    ) -> List[SoftwareBuild]:
        """Generates a collection of independent synthetic builds."""
        builds = []
        for i in range(num_builds):
            b = self.generate_synthetic_build(f"{prefix}_{i:04d}", seed=base_seed + i)
            builds.append(b)
        return builds


class RealWorldCIAdapter:
    """
    Transforms real-world CI datasets (e.g. ABB Paint Control, IOF/ROL)
    into SoftwareBuild instances for realistic RL evaluation.
    """

    @staticmethod
    def from_ci_dataframe(
        df: pd.DataFrame,
        cycle_id: int,
        budget_ratio: float = 0.45,
        estimated_features: int = 15,
        max_tests_limit: int = 50,
    ) -> Optional[SoftwareBuild]:
        """
        Converts a single CI cycle from PaintControl or IOF/ROL into a SoftwareBuild.
        """
        cycle_df = df[df["Cycle"] == cycle_id].copy()
        if len(cycle_df) < 5:
            return None

        # If too many tests in cycle, sample or take top execution costs to keep manageable
        if len(cycle_df) > max_tests_limit:
            cycle_df = cycle_df.head(max_tests_limit)

        test_names = cycle_df["Name"].tolist()
        num_tests = len(test_names)
        test_cases = list(range(num_tests))

        # Durations normalized to practical cost units (1.0 to 10.0 scale)
        durations = cycle_df["Duration"].values.astype(float)
        min_d = durations.min()
        max_d = durations.max()
        if max_d > min_d:
            norm_costs = 1.0 + 9.0 * ((durations - min_d) / (max_d - min_d))
        else:
            norm_costs = np.full(num_tests, 2.0)
        costs = [round(float(c), 2) for c in norm_costs]

        # In CI, Verdict=1 means test revealed a failure/bug
        verdicts = cycle_df["Verdict"].values
        hidden_bugs = set()
        bug_detection_map: Dict[int, Set[int]] = {t: set() for t in test_cases}

        bug_counter = 0
        for idx, v in enumerate(verdicts):
            if v == 1:
                hidden_bugs.add(bug_counter)
                bug_detection_map[idx].add(bug_counter)
                bug_counter += 1

        # Derive feature coverage pseudo-deterministically from test name hash
        coverage_map: Dict[int, Set[int]] = {}
        for idx, tname in enumerate(test_names):
            h = hash(str(tname)) % 100000
            rng = np.random.default_rng(abs(h))
            n_feats = int(rng.integers(1, 4))
            feats = set(rng.choice(estimated_features, size=n_feats, replace=False).tolist())
            coverage_map[idx] = feats

        total_cost = sum(costs)
        budget = max(1.0, round(total_cost * budget_ratio, 2))

        return SoftwareBuild(
            build_id=f"ci_cycle_{cycle_id}",
            test_cases=test_cases,
            costs=costs,
            coverage_map=coverage_map,
            hidden_bugs=hidden_bugs,
            bug_detection_map=bug_detection_map,
            budget=budget,
            total_features=estimated_features,
        )

    @classmethod
    def load_all_builds_from_df(
        cls, df: pd.DataFrame, cycles: List[int], budget_ratio: float = 0.45
    ) -> List[SoftwareBuild]:
        """Converts a list of CI cycle numbers into SoftwareBuild instances."""
        builds = []
        for c in cycles:
            b = cls.from_ci_dataframe(df, cycle_id=c, budget_ratio=budget_ratio)
            if b is not None and b.num_hidden_bugs > 0:
                builds.append(b)
        return builds
