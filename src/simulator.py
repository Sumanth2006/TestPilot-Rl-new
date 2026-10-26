"""
Simulator module for TestPilot-RL.
Generates synthetic software builds and transforms real CI logs into build scenarios.

Strict Information Boundary Principle:
- Ground truth (actual bugs, test-to-bug mappings, current build verdicts) is maintained
  strictly inside the Simulator/Build data structure.
- Historical features (`recent_verdicts`, `causal_fail_rates`, `co_failure_map`) are
  constructed EXCLUSIVELY from strictly preceding builds/cycles (`0 .. i-1`).
  When no prior builds exist (`i == 0`), all tests receive uniform uninformative priors.
"""

import zlib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
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
    - test_names: Optional mapping action_idx -> canonical test name
    - global_test_ids: Optional mapping action_idx -> global unique test index
    - recent_verdicts: Optional mapping action_idx -> list of prior verdicts (strictly from past builds)
    - causal_fail_rates: Optional mapping action_idx -> cumulative failure rate prior to this build
    - co_failure_map: Optional mapping action_idx -> set of co-failing action indices from prior builds
    """
    build_id: str
    test_cases: List[int]
    costs: List[float]
    coverage_map: Dict[int, Set[int]]
    hidden_bugs: Set[int]
    bug_detection_map: Dict[int, Set[int]]
    budget: float
    total_features: int
    test_names: Dict[int, str] = field(default_factory=dict)
    global_test_ids: Dict[int, int] = field(default_factory=dict)
    recent_verdicts: Dict[int, List[int]] = field(default_factory=dict)
    causal_fail_rates: Dict[int, float] = field(default_factory=dict)
    co_failure_map: Dict[int, Set[int]] = field(default_factory=dict)

    @property
    def num_tests(self) -> int:
        return len(self.test_cases)

    @property
    def total_cost(self) -> float:
        return float(sum(self.costs))

    @property
    def num_hidden_bugs(self) -> int:
        return len(self.hidden_bugs)

    def get_test_key(self, action: int) -> Any:
        """Return canonical test identifier (name or global ID) for history lookups."""
        if action in self.test_names:
            return self.test_names[action]
        if action in self.global_test_ids:
            return self.global_test_ids[action]
        return action

    def lookup_history(
        self,
        action: int,
        test_history: Optional[Dict[Any, float]] = None,
        default: float = 0.10,
    ) -> float:
        """
        Safely resolves historical failure rate for `action` without row-index mismatch.
        Checks canonical test name first, then global test ID, then local action ID.
        """
        if test_history:
            tname = self.test_names.get(action)
            if tname is not None and tname in test_history:
                return float(test_history[tname])
            gid = self.global_test_ids.get(action)
            if gid is not None and gid in test_history:
                return float(test_history[gid])
            if action in test_history and not self.test_names and not self.global_test_ids:
                return float(test_history[action])
            if action in test_history and len(test_history) == self.num_tests:
                return float(test_history[action])

        if action in self.causal_fail_rates:
            return float(self.causal_fail_rates[action])
        return float(default)

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
            "test_names": {str(k): v for k, v in self.test_names.items()},
            "global_test_ids": {str(k): int(v) for k, v in self.global_test_ids.items()},
            "recent_verdicts": {str(k): list(v) for k, v in self.recent_verdicts.items()},
            "causal_fail_rates": {str(k): float(v) for k, v in self.causal_fail_rates.items()},
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
            test_names={int(k): str(v) for k, v in data.get("test_names", {}).items()},
            global_test_ids={int(k): int(v) for k, v in data.get("global_test_ids", {}).items()},
            recent_verdicts={int(k): list(v) for k, v in data.get("recent_verdicts", {}).items()},
            causal_fail_rates={int(k): float(v) for k, v in data.get("causal_fail_rates", {}).items()},
        )


class BuildGenerator:
    """
    Generates synthetic software builds with realistic software engineering structure
    and ZERO information leakage from the current build's hidden bugs:
    - All historical features (`recent_verdicts`, `causal_fail_rates`, `co_failure_map`)
      are accumulated strictly from prior builds `0 .. i-1`.
    - Test costs vary by ±15% across builds, and test coverage has per-build variation
      so the agent must generalize across feature representations rather than memorizing static indices.
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

        arch_rng = np.random.default_rng(104729 + num_tests * 31 + num_features * 17)
        self.num_hotspots = max(2, num_features // 2)

        self.base_costs: List[float] = []
        self.base_coverage: Dict[int, Set[int]] = {}
        self.test_archetype: Dict[int, str] = {}
        self.primary_hotspot: Dict[int, int] = {}

        n_redundant = max(4, int(num_tests * 0.24))
        n_utility = max(4, int(num_tests * 0.23))
        n_core = max(6, int(num_tests * 0.40))

        for t in range(num_tests):
            if t < n_redundant:
                self.test_archetype[t] = "redundant"
                self.primary_hotspot[t] = 0
                self.base_coverage[t] = {0}
                self.base_costs.append(round(float(arch_rng.uniform(1.65, 2.15)), 2))
            elif t < n_redundant + n_utility:
                self.test_archetype[t] = "utility"
                self.primary_hotspot[t] = -1
                util_pool = list(range(self.num_hotspots, num_features))
                if len(util_pool) >= 3:
                    k = min(len(util_pool), 3)
                    feats = set(arch_rng.choice(util_pool, size=k, replace=False).tolist())
                else:
                    feats = {t % num_features}
                self.base_coverage[t] = feats
                self.base_costs.append(round(float(arch_rng.uniform(1.15, 1.60)), 2))
            elif t < n_redundant + n_utility + n_core:
                self.test_archetype[t] = "core"
                idx_c = t - n_redundant - n_utility
                hs = 1 + (idx_c % max(1, self.num_hotspots - 1))
                self.primary_hotspot[t] = hs
                extra_f = self.num_hotspots + (idx_c % max(1, num_features - self.num_hotspots))
                self.base_coverage[t] = {hs, extra_f}
                self.base_costs.append(round(float(arch_rng.uniform(1.50, 2.10)), 2))
            else:
                self.test_archetype[t] = "integration"
                hs = t % self.num_hotspots
                self.primary_hotspot[t] = hs
                self.base_coverage[t] = {hs, (hs + 1) % num_features}
                self.base_costs.append(round(float(arch_rng.uniform(3.30, 4.60)), 2))

    def generate_synthetic_build(
        self,
        build_id: str,
        seed: Optional[int] = None,
        prior_verdicts: Optional[Dict[int, List[int]]] = None,
        causal_rates: Optional[Dict[int, float]] = None,
        prior_co_failures: Optional[Dict[int, Set[int]]] = None,
        active_core_tests: Optional[List[int]] = None,
    ) -> SoftwareBuild:
        """
        Generates a single synthetic build.
        CRITICAL: `recent_verdicts`, `causal_fail_rates`, and `co_failure_map`
        are populated ONLY from `prior_verdicts`, `causal_rates`, and `prior_co_failures`
        (strictly past builds). When None (first build), uniform uninformative defaults
        are used so zero ground-truth information from the current build leaks.
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng

        test_cases = list(range(self.num_tests))
        costs = []
        for t in test_cases:
            # ±15% runtime cost variation across builds
            jitter = float(rng.uniform(0.85, 1.15))
            c = np.clip(self.base_costs[t] * jitter, self.cost_range[0], self.cost_range[1])
            costs.append(round(float(c), 2))

        # Per-build coverage variation (occasionally exercises an adjacent utility feature)
        coverage_map: Dict[int, Set[int]] = {}
        for t in test_cases:
            feats = set(self.base_coverage[t])
            if self.test_archetype[t] in ("utility", "integration") and rng.random() < 0.25:
                extra = int(rng.integers(self.num_hotspots, self.num_features))
                feats.add(extra)
            coverage_map[t] = feats

        num_bugs = int(rng.integers(self.min_bugs, self.max_bugs + 1))
        hidden_bugs = set(range(num_bugs))
        bug_detection_map: Dict[int, Set[int]] = {t: set() for t in test_cases}

        redundant_tests = [t for t in test_cases if self.test_archetype[t] == "redundant"]
        core_tests = [t for t in test_cases if self.test_archetype[t] == "core"]
        utility_tests = [t for t in test_cases if self.test_archetype[t] == "utility"]

        # Bug 0 occurs in Feature 0 in 90% of builds and is detected by redundant tests
        if rng.random() < 0.90:
            for rt in redundant_tests:
                bug_detection_map[rt].add(0)
        elif core_tests:
            bug_detection_map[int(rng.choice(core_tests))].add(0)

        chosen_core = active_core_tests if active_core_tests else rng.choice(
            core_tests, size=min(len(core_tests), max(1, num_bugs - 1)), replace=False
        ).tolist()

        for b_idx, bug_id in enumerate(range(1, num_bugs)):
            target_t = chosen_core[b_idx % len(chosen_core)]
            bug_detection_map[int(target_t)].add(bug_id)
            if b_idx == 0 and utility_tests and rng.random() < 0.35:
                bug_detection_map[int(rng.choice(utility_tests))].add(bug_id)

        # Strictly causal co-failure map (ONLY from past builds; empty if no past builds)
        co_failure_map: Dict[int, Set[int]] = {
            t: set(prior_co_failures.get(t, set())) if prior_co_failures else set()
            for t in test_cases
        }

        # Strictly causal recent verdicts (ONLY from past builds; empty list if no past builds)
        rec_verdicts: Dict[int, List[int]] = {
            t: list(prior_verdicts[t][:5]) if (prior_verdicts and t in prior_verdicts) else []
            for t in test_cases
        }

        # Strictly causal failure rates (ONLY from past builds; uniform 0.10 if no past builds)
        c_rates: Dict[int, float] = {
            t: float(causal_rates[t]) if (causal_rates and t in causal_rates) else 0.10
            for t in test_cases
        }

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
            test_names={t: f"SynTest_{t:02d}" for t in test_cases},
            global_test_ids={t: t for t in test_cases},
            recent_verdicts=rec_verdicts,
            causal_fail_rates=c_rates,
            co_failure_map=co_failure_map,
        )

    def generate_dataset(
        self, num_builds: int = 100, prefix: str = "build", base_seed: int = 1000
    ) -> List[SoftwareBuild]:
        """
        Generates a chronological sequence of synthetic builds where all historical
        signals (`prior_verdicts`, `causal_rates`, `prior_co_failures`) are updated
        strictly AFTER each build completes.
        """
        builds = []
        seq_rng = np.random.default_rng(base_seed)
        core_tests = [t for t in range(self.num_tests) if self.test_archetype[t] == "core"]

        prior_verdicts: Dict[int, List[int]] = {t: [] for t in range(self.num_tests)}
        fail_counts: Dict[int, int] = {t: 0 for t in range(self.num_tests)}
        co_fail_counts: Dict[int, Dict[int, int]] = {t: {} for t in range(self.num_tests)}

        active_core = seq_rng.choice(core_tests, size=min(6, len(core_tests)), replace=False).tolist()

        for i in range(num_builds):
            if i > 0 and i % 5 == 0 and len(core_tests) > len(active_core):
                inactive = [t for t in core_tests if t not in active_core]
                active_core.pop(0)
                active_core.append(int(seq_rng.choice(inactive)))

            # Strictly causal failure rates from builds 0 .. i-1 (uniform 0.10 at i == 0)
            causal_rates = {
                t: (fail_counts[t] / float(i)) if i > 0 else 0.10
                for t in range(self.num_tests)
            }
            # Strictly causal co-failure links from builds 0 .. i-1 (filtered by conditional lift)
            prior_co_failures = {
                t: {
                    other
                    for other, cnt in co_fail_counts[t].items()
                    if cnt >= 2 and (cnt / max(1, fail_counts[t])) >= 0.35
                }
                for t in range(self.num_tests)
            }

            b = self.generate_synthetic_build(
                f"{prefix}_{i:04d}",
                seed=base_seed + i,
                prior_verdicts=prior_verdicts,
                causal_rates=causal_rates,
                prior_co_failures=prior_co_failures,
                active_core_tests=list(active_core),
            )
            builds.append(b)

            # Update causal history strictly AFTER build i is created
            failed_in_build = []
            for t in b.test_cases:
                failed = 1 if len(b.bug_detection_map.get(t, set())) > 0 else 0
                prior_verdicts[t].insert(0, failed)
                if len(prior_verdicts[t]) > 6:
                    prior_verdicts[t].pop()
                fail_counts[t] += failed
                if failed == 1:
                    failed_in_build.append(t)

            if len(failed_in_build) > 1:
                for t1 in failed_in_build:
                    for t2 in failed_in_build:
                        if t1 != t2:
                            co_fail_counts[t1][t2] = co_fail_counts[t1].get(t2, 0) + 1

        return builds


class RealWorldCIAdapter:
    """
    Transforms real-world CI datasets (e.g. ABB Paint Control, IOF/ROL)
    into SoftwareBuild instances for retrospective RL evaluation.

    Methodological Notes:
    - Each failing test execution (`Verdict == 1`) is treated as a failed-test
      detection target (retrospective failure retention / recall).
    - Because the public RETECS logs do not include source-code coverage matrices,
      `coverage_map` uses a deterministic test-name token grouping proxy (`name_token_proxy`).
    """

    @staticmethod
    def _deterministic_features(tname: str, estimated_features: int) -> Set[int]:
        """Derives deterministic name-token proxy groups from test name identifiers."""
        tokens = str(tname).replace("-", "_").split("_")
        feats = set()
        prefix = "_".join(tokens[:2]) if len(tokens) >= 2 else str(tname)
        h_prefix = zlib.crc32(prefix.encode("utf-8")) % estimated_features
        feats.add(int(h_prefix))

        h_full = zlib.crc32(str(tname).encode("utf-8"))
        n_extra = int((h_full % 2) + 1)
        for k in range(n_extra):
            feats.add(int((h_full // (k + 3)) % estimated_features))
        return feats

    @classmethod
    def from_ci_dataframe(
        cls,
        df: pd.DataFrame,
        cycle_id: int,
        budget_ratio: float = 0.45,
        estimated_features: int = 15,
        max_tests_limit: int = 100,
        _precomputed_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[SoftwareBuild]:
        cycle_df = df[df["Cycle"] == cycle_id].copy()
        if len(cycle_df) < 5:
            return None

        if len(cycle_df) > max_tests_limit:
            cycle_df = cycle_df.head(max_tests_limit)

        test_names_list = cycle_df["Name"].astype(str).tolist()
        num_tests = len(test_names_list)
        test_cases = list(range(num_tests))

        if _precomputed_context and "name_to_idx" in _precomputed_context:
            name_to_idx = _precomputed_context["name_to_idx"]
        else:
            all_names = df["Name"].astype(str).unique().tolist()
            name_to_idx = {name: idx for idx, name in enumerate(all_names)}

        if _precomputed_context and "prior_verdicts_by_name" in _precomputed_context:
            prior_verdicts_map = _precomputed_context["prior_verdicts_by_name"]
            prior_rates_map = _precomputed_context["prior_rates_by_name"]
            co_fail_names_map = _precomputed_context.get("co_fail_names_map", {})
        else:
            prior_df = df[df["Cycle"] < cycle_id]
            prior_verdicts_map = {}
            prior_rates_map = {}
            co_fail_names_map = {}
            if len(prior_df) > 0:
                for name, grp in prior_df.groupby("Name"):
                    v_list = grp["Verdict"].astype(int).tolist()[::-1]
                    prior_verdicts_map[str(name)] = v_list[:10]
                    prior_rates_map[str(name)] = float(np.mean(v_list))

        durations = cycle_df["Duration"].values.astype(float)
        min_d = float(durations.min())
        max_d = float(durations.max())
        if max_d > min_d:
            norm_costs = 1.0 + 9.0 * ((durations - min_d) / (max_d - min_d))
        else:
            norm_costs = np.full(num_tests, 2.0)
        costs = [round(float(c), 2) for c in norm_costs]

        verdicts = cycle_df["Verdict"].values.astype(int)
        hidden_bugs = set()
        bug_detection_map: Dict[int, Set[int]] = {t: set() for t in test_cases}

        bug_counter = 0
        for idx, v in enumerate(verdicts):
            if v == 1:
                hidden_bugs.add(bug_counter)
                bug_detection_map[idx].add(bug_counter)
                bug_counter += 1

        coverage_map: Dict[int, Set[int]] = {}
        test_names_dict: Dict[int, str] = {}
        global_ids_dict: Dict[int, int] = {}
        recent_verdicts_dict: Dict[int, List[int]] = {}
        causal_rates_dict: Dict[int, float] = {}

        raw_last_results = cycle_df["LastResults"].tolist() if "LastResults" in cycle_df.columns else [[] for _ in range(num_tests)]

        for idx, tname in enumerate(test_names_list):
            test_names_dict[idx] = tname
            global_ids_dict[idx] = int(name_to_idx.get(tname, idx))
            coverage_map[idx] = cls._deterministic_features(tname, estimated_features)

            p_verdicts = list(prior_verdicts_map.get(tname, []))
            csv_last = raw_last_results[idx] if isinstance(raw_last_results[idx], list) else []
            combined_recent = p_verdicts if p_verdicts else [int(x) for x in csv_last[:10]]
            recent_verdicts_dict[idx] = combined_recent[:8]

            if tname in prior_rates_map:
                causal_rates_dict[idx] = float(prior_rates_map[tname])
            elif combined_recent:
                causal_rates_dict[idx] = float(np.mean(combined_recent))
            else:
                causal_rates_dict[idx] = 0.05

        # Map each test name to ALL action indices with that name in this cycle
        name_to_locals: Dict[str, List[int]] = {}
        for idx, tname in enumerate(test_names_list):
            name_to_locals.setdefault(tname, []).append(idx)

        co_failure_map: Dict[int, Set[int]] = {t: set() for t in test_cases}
        for idx, tname in enumerate(test_names_list):
            co_names = co_fail_names_map.get(tname, set())
            linked_indices: Set[int] = set()
            for cn in co_names:
                if cn in name_to_locals:
                    for target_idx in name_to_locals[cn]:
                        if target_idx != idx:
                            linked_indices.add(target_idx)
            co_failure_map[idx] = linked_indices

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
            test_names=test_names_dict,
            global_test_ids=global_ids_dict,
            recent_verdicts=recent_verdicts_dict,
            causal_fail_rates=causal_rates_dict,
            co_failure_map=co_failure_map,
        )

    @classmethod
    def load_all_builds_from_df(
        cls,
        df: pd.DataFrame,
        cycles: List[int],
        budget_ratio: float = 0.45,
        include_zero_failure_cycles: bool = False,
    ) -> List[SoftwareBuild]:
        all_names = df["Name"].astype(str).unique().tolist()
        name_to_idx = {name: idx for idx, name in enumerate(all_names)}

        target_set = set(cycles)
        all_sorted_cycles = sorted(df["Cycle"].unique())

        prior_verdicts_by_name: Dict[str, List[int]] = {}
        fail_counts_by_name: Dict[str, int] = {}
        exec_counts_by_name: Dict[str, int] = {}
        co_fail_counts: Dict[str, Dict[str, int]] = {}

        builds_by_cycle: Dict[int, SoftwareBuild] = {}

        for c in all_sorted_cycles:
            cycle_df = df[df["Cycle"] == c]
            if c in target_set:
                prior_rates = {
                    name: fail_counts_by_name.get(name, 0) / max(1, exec_counts_by_name[name])
                    for name in exec_counts_by_name
                }
                # High-precision conditional lift co-failure links:
                # Requires >= 2 co-failures, conditional prob P(other | name) >= 0.40, and positive lift > 1.3x
                co_fail_names_map = {
                    name: {
                        other
                        for other, cnt in neighbors.items()
                        if cnt >= 2
                        and (cnt / max(1, fail_counts_by_name.get(name, 1))) >= 0.40
                        and (cnt / max(1, fail_counts_by_name.get(name, 1)))
                        > 1.30 * prior_rates.get(other, 0.05)
                    }
                    for name, neighbors in co_fail_counts.items()
                }
                ctx = {
                    "name_to_idx": name_to_idx,
                    "prior_verdicts_by_name": prior_verdicts_by_name,
                    "prior_rates_by_name": prior_rates,
                    "co_fail_names_map": co_fail_names_map,
                }
                b = cls.from_ci_dataframe(
                    df, cycle_id=c, budget_ratio=budget_ratio, _precomputed_context=ctx
                )
                if b is not None and (include_zero_failure_cycles or b.num_hidden_bugs > 0):
                    builds_by_cycle[c] = b

            failed_in_cycle = []
            for tname, v in zip(cycle_df["Name"].astype(str).values, cycle_df["Verdict"].astype(int).values):
                if tname not in prior_verdicts_by_name:
                    prior_verdicts_by_name[tname] = []
                prior_verdicts_by_name[tname].insert(0, int(v))
                if len(prior_verdicts_by_name[tname]) > 10:
                    prior_verdicts_by_name[tname].pop()
                exec_counts_by_name[tname] = exec_counts_by_name.get(tname, 0) + 1
                if v == 1:
                    fail_counts_by_name[tname] = fail_counts_by_name.get(tname, 0) + 1
                    failed_in_cycle.append(tname)

            unique_failed = list(set(failed_in_cycle))
            if len(unique_failed) > 1:
                for f1 in unique_failed:
                    if f1 not in co_fail_counts:
                        co_fail_counts[f1] = {}
                    for f2 in unique_failed:
                        if f1 != f2:
                            co_fail_counts[f1][f2] = co_fail_counts[f1].get(f2, 0) + 1

        return [builds_by_cycle[c] for c in cycles if c in builds_by_cycle]