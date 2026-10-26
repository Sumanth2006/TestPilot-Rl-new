"""
Evaluation module for TestPilot-RL.

Evaluates Reinforcement Learning policies (Dueling DQN and Linear Q-Learning)
against Random and Greedy baselines (including symmetric Greedy Recency/Cost) under:
- Identical held-out test scenarios
- Strictly enforced equal budgets
- Multi-seed training & evaluation with causal online adaptation
- Full-suite denominator N for APFD and NAPFD comparability across policies
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

from src.simulator import BuildGenerator, RealWorldCIAdapter, SoftwareBuild
from src.environment import TestingEnvironment
from src.q_learning import DuelingDQNAgent, LinearQLearningAgent, TabularQLearningAgent
from src.baselines import (
    RandomBaseline,
    GreedyCoverageCostBaseline,
    GreedyFailureRateBaseline,
    GreedyRecencyCostBaseline,
)
from src.metrics import compute_apfd, compute_napfd, aggregate_experiment_results, compute_paired_cycle_comparisons
from src.data_loader import load_ci_dataset, split_ci_cycles
from src.train import build_canonical_ci_history


def evaluate_policy_on_build(
    policy: Any,
    env: TestingEnvironment,
    build: SoftwareBuild,
    is_rl: bool = False,
    online_adapt: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates a single policy on one software build until budget depletion or termination.
    When `online_adapt=True`, updates causal history after the build for any policy
    implementing `adapt_online` (both RL agents and `GreedyRecencyCostBaseline`),
    using only the observed outcomes of the executed tests.
    """
    state, info = env.reset(build)
    execution_order = []
    total_reward = 0.0

    while True:
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            break

        if is_rl:
            action = policy.select_action(state, legal_actions, explore=False, env=env)
        else:
            action = policy.select_action(env)

        execution_order.append(action)
        next_state, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        state = next_state

        if terminated:
            break

    if online_adapt and hasattr(policy, "adapt_online"):
        policy.adapt_online(build, execution_order)

    final_info = env._get_info()
    # Use full suite size N = build.num_tests so APFD and NAPFD are on the exact same scale across policies
    apfd_score = compute_apfd(
        execution_order, build.bug_detection_map, build.num_hidden_bugs, total_tests=build.num_tests
    )
    napfd_score = compute_napfd(
        execution_order, build.bug_detection_map, build.num_hidden_bugs, total_tests=build.num_tests
    )

    return {
        "build_id": build.build_id,
        "bug_detection_rate": final_info["bug_detection_rate"],
        "unique_bugs_detected": final_info["unique_bugs_detected"],
        "total_hidden_bugs": final_info["total_hidden_bugs"],
        "coverage_rate": final_info["coverage_rate"],
        "budget_consumed": final_info["budget_consumed"],
        "total_budget": final_info["total_budget"],
        "budget_consumed_ratio": final_info["budget_consumed_ratio"],
        "bugs_per_cost_unit": final_info["bugs_per_cost_unit"],
        "tests_executed_count": final_info["tests_executed_count"],
        "total_tests_available": final_info["total_tests_available"],
        "apfd": apfd_score,
        "napfd": napfd_score,
        "eval_reward": total_reward,
    }


def run_benchmark_comparison(
    agent: Any,
    test_builds: List[SoftwareBuild],
    seeds: List[int] = [42, 101, 777],
    test_history: Optional[Dict[Any, float]] = None,
    extra_agents: Optional[List[Tuple[str, Any]]] = None,
    trained_agents_by_seed: Optional[Dict[int, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Runs fair, symmetric head-to-head comparison between RL agent(s) and baselines
    (including Greedy Recency/Cost with identical online updates) across held-out
    test scenarios and multiple seeds.
    """
    results = []
    env = TestingEnvironment(build=test_builds[0])
    base_history = dict(test_history) if test_history else {}

    for seed in seeds:
        if trained_agents_by_seed and seed in trained_agents_by_seed:
            seed_linear = trained_agents_by_seed[seed].get("Q-Learning", agent)
            seed_extras = [
                (k, v) for k, v in trained_agents_by_seed[seed].items() if k != "Q-Learning"
            ]
        else:
            seed_linear = agent
            seed_extras = extra_agents or []

        if hasattr(seed_linear, "set_test_history"):
            seed_linear.set_test_history(dict(base_history))
        for _, ext_ag in seed_extras:
            if hasattr(ext_ag, "set_test_history"):
                ext_ag.set_test_history(dict(base_history))

        random_baseline = RandomBaseline(seed=seed)
        greedy_cov = GreedyCoverageCostBaseline()
        greedy_fail = GreedyFailureRateBaseline(test_history=dict(base_history))
        greedy_recency = GreedyRecencyCostBaseline(test_history=dict(base_history))

        policies: List[Tuple[str, Any, bool]] = [
            ("Random", random_baseline, False),
            ("Greedy (Coverage/Cost)", greedy_cov, False),
            ("Greedy (Failure/Cost)", greedy_fail, False),
            ("Greedy (Recency/Cost)", greedy_recency, False),
            ("Q-Learning", seed_linear, True),
        ]
        for ext_name, ext_ag in seed_extras:
            policies.append((ext_name, ext_ag, True))

        for build in test_builds:
            for strat_name, pol, is_rl in policies:
                metrics = evaluate_policy_on_build(pol, env, build, is_rl=is_rl, online_adapt=True)
                metrics["strategy"] = strat_name
                metrics["eval_seed"] = seed
                results.append(metrics)

    if hasattr(agent, "set_test_history"):
        agent.set_test_history(dict(base_history))
    if extra_agents:
        for _, ext_ag in extra_agents:
            if hasattr(ext_ag, "set_test_history"):
                ext_ag.set_test_history(dict(base_history))

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TestPilot-RL vs Baselines")
    parser.add_argument("--model_path", type=str, default="models/q_table.npy")
    parser.add_argument("--num_test_builds", type=int, default=120)
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "paintcontrol", "iofrol"])
    parser.add_argument("--raw_output", type=str, default="results/scratch/raw_results.csv")
    parser.add_argument("--summary_output", type=str, default="results/scratch/summary.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    test_history: Dict[Any, float] = {}
    if args.dataset == "synthetic":
        bg_train = BuildGenerator(num_tests=30, num_features=12, min_bugs=5, max_bugs=8, budget_ratio=0.45, seed=args.seed)
        bg_test = BuildGenerator(num_tests=30, num_features=12, min_bugs=5, max_bugs=8, budget_ratio=0.45, seed=99)
        train_builds = bg_train.generate_dataset(num_builds=80, prefix="train_build", base_seed=args.seed)
        test_builds = bg_test.generate_dataset(num_builds=args.num_test_builds, prefix="heldout_build", base_seed=5000)
        for t in range(30):
            rate = sum(1 for b in train_builds if len(b.bug_detection_map.get(t, set())) > 0) / max(1, len(train_builds))
            test_history[t] = rate
            test_history[f"SynTest_{t:02d}"] = rate
    else:
        df = load_ci_dataset(args.dataset)
        train_cycles, test_cycles = split_ci_cycles(df, train_ratio=0.7)
        test_builds = RealWorldCIAdapter.load_all_builds_from_df(df, test_cycles, budget_ratio=0.45)
        test_history = build_canonical_ci_history(df, train_cycles)

    agent = LinearQLearningAgent(num_actions=max(b.num_tests for b in test_builds), test_history=test_history)
    if os.path.exists(args.model_path):
        agent.load(args.model_path)
        print(f"Loaded trained model from {args.model_path}")
    else:
        print(f"Warning: {args.model_path} not found. Running with untrained agent.")

    raw_df = run_benchmark_comparison(agent, test_builds, seeds=[42, 101, 777], test_history=test_history)
    summary_df = aggregate_experiment_results(raw_df)

    os.makedirs(os.path.dirname(args.raw_output) or ".", exist_ok=True)
    raw_df.to_csv(args.raw_output, index=False)
    summary_df.to_csv(args.summary_output, index=False)

    print("\n=== EVALUATION SUMMARY TABLE ===")
    print(summary_df[["strategy", "bug_detection_rate_mean", "bug_detection_rate_std",
                      "coverage_rate_mean", "apfd_mean", "napfd_mean", "bugs_per_cost_unit_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()