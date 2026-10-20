"""
Evaluation module for TestPilot-RL.

Evaluates Q-Learning vs. Random and Greedy baselines under:
- Identical held-out test scenarios (at least 120 builds)
- Strictly enforced equal budgets
- Multi-seed evaluation
- Comprehensive metric collection (BDR, Coverage, Efficiency, APFD)
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from src.simulator import BuildGenerator, RealWorldCIAdapter, SoftwareBuild
from src.environment import TestingEnvironment
from src.q_learning import LinearQLearningAgent, TabularQLearningAgent
from src.baselines import RandomBaseline, GreedyCoverageCostBaseline, GreedyFailureRateBaseline
from src.metrics import compute_apfd, aggregate_experiment_results
from src.data_loader import load_ci_dataset, split_ci_cycles


def evaluate_policy_on_build(
    policy: Any,
    env: TestingEnvironment,
    build: SoftwareBuild,
    is_rl: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates a single policy on one software build until budget depletion or termination.
    """
    state, info = env.reset(build)
    execution_order = []
    total_reward = 0.0

    while True:
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            break

        if is_rl:
            # Exploitation only (no exploration during evaluation)
            action = policy.select_action(state, legal_actions, explore=False, env=env)
        else:
            action = policy.select_action(env)

        execution_order.append(action)
        next_state, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        state = next_state

        if terminated:
            break

    final_info = env._get_info()
    apfd_score = compute_apfd(execution_order, build.bug_detection_map, build.num_hidden_bugs)

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
        "eval_reward": total_reward,
    }


def run_benchmark_comparison(
    agent: Any,
    test_builds: List[SoftwareBuild],
    seeds: List[int] = [42, 101, 777],
    test_history: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """
    Runs fair, head-to-head comparison between Q-Learning and baselines
    across all held-out test scenarios and multiple evaluation seeds.
    """
    results = []
    env = TestingEnvironment(build=test_builds[0])
    if test_history is not None:
        agent.set_test_history(test_history)

    for seed in seeds:
        random_baseline = RandomBaseline(seed=seed)
        greedy_cov = GreedyCoverageCostBaseline()
        greedy_fail = GreedyFailureRateBaseline(test_history=test_history)

        policies = [
            ("Random", random_baseline, False),
            ("Greedy (Coverage/Cost)", greedy_cov, False),
            ("Greedy (Failure/Cost)", greedy_fail, False),
            ("Q-Learning", agent, True),
        ]

        for build in test_builds:
            for strat_name, pol, is_rl in policies:
                metrics = evaluate_policy_on_build(pol, env, build, is_rl=is_rl)
                metrics["strategy"] = strat_name
                metrics["eval_seed"] = seed
                results.append(metrics)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TestPilot-RL vs Baselines")
    parser.add_argument("--model_path", type=str, default="models/scratch_model.npy")
    parser.add_argument("--num_test_builds", type=int, default=120)
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "paintcontrol", "iofrol"])
    parser.add_argument("--raw_output", type=str, default="results/scratch/raw_results.csv")
    parser.add_argument("--summary_output", type=str, default="results/scratch/summary.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    test_history = None
    if args.dataset == "synthetic":
        bg = BuildGenerator(num_tests=30, num_features=12, min_bugs=5, max_bugs=8, budget_ratio=0.45, seed=args.seed + 1000)
        test_builds = bg.generate_dataset(num_builds=args.num_test_builds, prefix="heldout_build", base_seed=args.seed + 5000)
    else:
        df = load_ci_dataset(args.dataset)
        _, test_cycles = split_ci_cycles(df, train_ratio=0.7)
        test_builds = RealWorldCIAdapter.load_all_builds_from_df(df, test_cycles, budget_ratio=0.45)
        test_history = df.groupby("Name")["Verdict"].mean().to_dict()
        all_names = df["Name"].unique().tolist()
        name_to_idx = {name: idx for idx, name in enumerate(all_names)}
        test_history = {name_to_idx[n]: v for n, v in test_history.items() if n in name_to_idx}

    agent = LinearQLearningAgent(num_actions=max(b.num_tests for b in test_builds))
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
                        "coverage_rate_mean", "apfd_mean", "bugs_per_cost_unit_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
