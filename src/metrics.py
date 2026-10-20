"""
Scientific metrics and statistical evaluation module for TestPilot-RL.

Implements all primary and secondary evaluation metrics specified in the project plan:
1. Bug Detection Rate (BDR): unique bugs detected / total hidden bugs.
2. Unique Bugs Detected: absolute count discovered per episode.
3. Feature Coverage: unique features covered / total features.
4. Budget Consumed: actual testing cost used under budget limit.
5. Bug Detection Efficiency: unique bugs detected per unit of execution cost.
6. APFD (Average Percentage of Faults Detected): standard SE benchmark metric.
7. Statistical aggregations: mean, std dev, 95% confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any


def compute_apfd(execution_order: List[int], bug_detection_map: Dict[int, Any], total_bugs: int) -> float:
    """
    Computes Average Percentage of Faults Detected (APFD) for an executed test sequence.
    APFD = 1 - (sum(TF_i) / (n * m)) + (1 / (2 * n))
    where:
        n = number of tests executed
        m = total number of faults/bugs
        TF_i = position in execution order of first test detecting fault i
    """
    n = len(execution_order)
    m = total_bugs
    if n == 0 or m == 0:
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

    # For undetected faults, assign penalty position n + 1
    sum_tf = sum(first_detection_pos.values()) + (m - detected_count) * (n + 1)
    apfd = 1.0 - (sum_tf / (n * m)) + (1.0 / (2.0 * n))
    return float(np.clip(apfd, 0.0, 1.0))


def aggregate_experiment_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates raw evaluation results by strategy across independent scenarios/seeds.
    Computes mean, std, and 95% confidence intervals.
    """
    metrics = [
        "bug_detection_rate",
        "unique_bugs_detected",
        "coverage_rate",
        "budget_consumed",
        "bugs_per_cost_unit",
        "apfd",
    ]

    summary_rows = []
    strategies = raw_df["strategy"].unique()

    for strat in strategies:
        strat_df = raw_df[raw_df["strategy"] == strat]
        n_samples = len(strat_df)

        row = {"strategy": strat, "num_evaluations": n_samples}
        for m in metrics:
            if m in strat_df.columns:
                mean_val = float(strat_df[m].mean())
                std_val = float(strat_df[m].std(ddof=1)) if n_samples > 1 else 0.0
                ci95 = 1.96 * (std_val / np.sqrt(n_samples)) if n_samples > 1 else 0.0

                row[f"{m}_mean"] = round(mean_val, 4)
                row[f"{m}_std"] = round(std_val, 4)
                row[f"{m}_ci95"] = round(ci95, 4)

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)
