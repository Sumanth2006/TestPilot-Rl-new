"""
Scientific metrics and statistical evaluation module for TestPilot-RL.

Implements all primary and secondary evaluation metrics:
1. Bug Detection Rate (BDR): unique bugs detected / total hidden bugs (failing-test recall on CI logs).
2. Unique Bugs Detected: absolute count discovered per episode.
3. Feature Coverage: unique features covered / total features.
4. Budget Consumed: actual testing cost used under budget limit.
5. Bug Detection Efficiency: unique bugs detected per unit of execution cost.
6. NAPFD (Normalized APFD): Primary ordering metric (Spieker et al., ISSTA 2017) using
   full test suite size N so policies executing different numbers of tests are compared on an identical scale.
7. APFD (Budget-Penalized APFD): Full-suite denominator N where undetected faults receive penalty rank N + 1.
8. Cycle-level statistical aggregations & paired difference confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


def compute_apfd(
    execution_order: List[int],
    bug_detection_map: Dict[int, Any],
    total_bugs: int,
    total_tests: Optional[int] = None,
) -> float:
    """
    Computes Average Percentage of Faults Detected (APFD) across a test suite of size N.
    To ensure comparability across policies that execute different numbers of tests under a budget:
        N = total_tests (full available suite size in the build, or len(execution_order) if omitted)
        m = total_bugs
        TF_i = 1-based execution rank of the first test detecting fault i (or N + 1 if undetected)
        APFD = 1 - (sum(TF_i) / (N * m)) + (1 / (2 * N))
    """
    n_exec = len(execution_order)
    N = max(n_exec, int(total_tests)) if total_tests is not None else n_exec
    m = total_bugs
    if n_exec == 0 or N == 0 or m == 0:
        return 0.0

    first_detection_pos = {}
    for pos, test_id in enumerate(execution_order, start=1):
        detected_bugs = bug_detection_map.get(test_id, set())
        for b in detected_bugs:
            if b not in first_detection_pos:
                first_detection_pos[b] = pos

    detected_count = len(first_detection_pos)
    if detected_count == 0:
        return 0.0

    # Undetected faults receive penalty position N + 1 (full suite size + 1)
    sum_tf = sum(first_detection_pos.values()) + (m - detected_count) * (N + 1)
    apfd = 1.0 - (sum_tf / float(N * m)) + (1.0 / float(2 * N))
    return float(np.clip(apfd, 0.0, 1.0))


def compute_napfd(
    execution_order: List[int],
    bug_detection_map: Dict[int, Any],
    total_bugs: int,
    total_tests: Optional[int] = None,
) -> float:
    """
    Computes Normalized APFD (NAPFD) introduced in RETECS (Spieker et al., ISSTA 2017)
    for budget-constrained test case selection and prioritization:
        NAPFD = p - (sum_{detected i} TF_i) / (N * m) + p / (2 * N)
    where p = detected_bugs / total_bugs and N is the total available tests in the build
    (or number of executed tests if total_tests is None).
    """
    n_exec = len(execution_order)
    N = max(n_exec, int(total_tests)) if total_tests is not None else n_exec
    m = total_bugs
    if n_exec == 0 or N == 0 or m == 0:
        return 0.0

    first_detection_pos = {}
    for pos, test_id in enumerate(execution_order, start=1):
        detected_bugs = bug_detection_map.get(test_id, set())
        for b in detected_bugs:
            if b not in first_detection_pos:
                first_detection_pos[b] = pos

    detected_count = len(first_detection_pos)
    if detected_count == 0:
        return 0.0

    p = detected_count / float(m)
    sum_tf = sum(first_detection_pos.values())
    napfd = p - (sum_tf / float(N * m)) + (p / float(2 * N))
    return float(np.clip(napfd, 0.0, 1.0))


def aggregate_experiment_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates raw evaluation results by strategy.
    To avoid pseudoreplication across repeated random seeds on the same build/cycle,
    confidence intervals are computed across unique build-level means when build_id is present,
    and across training seeds when train_seed is present.
    """
    metrics = [
        "bug_detection_rate",
        "unique_bugs_detected",
        "coverage_rate",
        "budget_consumed",
        "bugs_per_cost_unit",
        "apfd",
        "napfd",
    ]

    summary_rows = []
    strategies = raw_df["strategy"].unique()

    for strat in strategies:
        strat_df = raw_df[raw_df["strategy"] == strat]
        n_evals = len(strat_df)

        if "build_id" in strat_df.columns and strat_df["build_id"].nunique() > 1:
            build_grouped = strat_df.groupby("build_id")
            n_independent = int(strat_df["build_id"].nunique())
        else:
            build_grouped = None
            n_independent = n_evals

        row = {
            "strategy": strat,
            "num_evaluations": n_evals,
            "num_unique_builds": n_independent,
        }
        for m in metrics:
            if m in strat_df.columns:
                if build_grouped is not None:
                    b_means = build_grouped[m].mean()
                    mean_val = float(b_means.mean())
                    std_val = float(b_means.std(ddof=1)) if n_independent > 1 else 0.0
                    ci95 = 1.96 * (std_val / np.sqrt(n_independent)) if n_independent > 1 else 0.0
                else:
                    mean_val = float(strat_df[m].mean())
                    std_val = float(strat_df[m].std(ddof=1)) if n_independent > 1 else 0.0
                    ci95 = 1.96 * (std_val / np.sqrt(n_independent)) if n_independent > 1 else 0.0

                row[f"{m}_mean"] = round(mean_val, 4)
                row[f"{m}_std"] = round(std_val, 4)
                row[f"{m}_ci95"] = round(ci95, 4)

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def compute_paired_cycle_comparisons(
    raw_df: pd.DataFrame,
    reference_strategy: str = "Dueling DQN (Online RL)",
) -> pd.DataFrame:
    """
    Computes paired cycle-by-cycle differences:
        d_i = Metric(reference, build_i) - Metric(baseline, build_i)
    averaged across seeds per build, reporting paired mean difference, 95% CI,
    and win/tie/loss counts across held-out builds.
    """
    if "build_id" not in raw_df.columns:
        return pd.DataFrame()

    strategies = list(raw_df["strategy"].unique())
    if reference_strategy not in strategies:
        rl_candidates = [s for s in strategies if "DQN" in s or "Q-Learning" in s]
        if not rl_candidates:
            return pd.DataFrame()
        reference_strategy = rl_candidates[0]

    pivot_bdr = raw_df.groupby(["build_id", "strategy"])["bug_detection_rate"].mean().unstack()
    pivot_apfd = raw_df.groupby(["build_id", "strategy"])["apfd"].mean().unstack()
    pivot_napfd = (
        raw_df.groupby(["build_id", "strategy"])["napfd"].mean().unstack()
        if "napfd" in raw_df.columns
        else pivot_apfd
    )

    rows = []
    for strat in strategies:
        if strat == reference_strategy or strat not in pivot_bdr.columns:
            continue

        diff_bdr = (pivot_bdr[reference_strategy] - pivot_bdr[strat]).dropna()
        diff_apfd = (pivot_apfd[reference_strategy] - pivot_apfd[strat]).dropna()
        diff_napfd = (pivot_napfd[reference_strategy] - pivot_napfd[strat]).dropna()
        n_b = len(diff_bdr)
        if n_b == 0:
            continue

        mean_d_bdr = float(diff_bdr.mean())
        std_d_bdr = float(diff_bdr.std(ddof=1)) if n_b > 1 else 0.0
        ci_d_bdr = 1.96 * (std_d_bdr / np.sqrt(n_b)) if n_b > 1 else 0.0

        mean_d_apfd = float(diff_apfd.mean())
        std_d_apfd = float(diff_apfd.std(ddof=1)) if n_b > 1 else 0.0
        ci_d_apfd = 1.96 * (std_d_apfd / np.sqrt(n_b)) if n_b > 1 else 0.0

        mean_d_napfd = float(diff_napfd.mean())
        std_d_napfd = float(diff_napfd.std(ddof=1)) if n_b > 1 else 0.0
        ci_d_napfd = 1.96 * (std_d_napfd / np.sqrt(n_b)) if n_b > 1 else 0.0

        wins = int((diff_bdr > 1e-6).sum())
        ties = int((np.abs(diff_bdr) <= 1e-6).sum())
        losses = int((diff_bdr < -1e-6).sum())

        rows.append({
            "reference": reference_strategy,
            "compared_against": strat,
            "num_paired_builds": n_b,
            "paired_bdr_diff_mean": round(mean_d_bdr, 4),
            "paired_bdr_diff_ci95": round(ci_d_bdr, 4),
            "paired_apfd_diff_mean": round(mean_d_apfd, 4),
            "paired_apfd_diff_ci95": round(ci_d_apfd, 4),
            "paired_napfd_diff_mean": round(mean_d_napfd, 4),
            "paired_napfd_diff_ci95": round(ci_d_napfd, 4),
            "builds_won": wins,
            "builds_tied": ties,
            "builds_lost": losses,
        })

    return pd.DataFrame(rows)