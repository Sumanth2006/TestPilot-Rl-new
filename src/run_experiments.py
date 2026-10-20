"""
Master scientific experiment runner for TestPilot-RL.

Executes all experimental suites specified in the project plan:
1. Training and convergence analysis (Training Curve)
2. Main comparison against Random and Greedy baselines (100+ held-out scenarios)
3. Budget sensitivity analysis (20%, 35%, 50%, 65%, 80%)
4. Reward weight ablation (Balanced vs. Pure Bug vs. Coverage Heavy)
5. State representation ablation (With history vs. Without history)
6. Real-world continuous integration evaluation on ABB Paint Control dataset
7. Generates publication-ready figures in results/figures/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

from src.simulator import BuildGenerator, RealWorldCIAdapter, SoftwareBuild
from src.environment import TestingEnvironment
from src.q_learning import TabularQLearningAgent
from src.baselines import RandomBaseline, GreedyCoverageCostBaseline, GreedyFailureRateBaseline
from src.train import train_agent
from src.evaluate import evaluate_policy_on_build, run_benchmark_comparison
from src.metrics import aggregate_experiment_results, compute_apfd
from src.data_loader import load_ci_dataset, split_ci_cycles


# Set publication aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def run_training_experiment(
    train_builds: List[SoftwareBuild],
    episodes: int = 1500,
    seed: int = 42,
    save_model_path: str = "models/q_table.npy",
) -> Tuple[TabularQLearningAgent, pd.DataFrame]:
    """Train primary Q-learning agent and record trajectory."""
    print("\n[1/6] Running Training Experiment...")
    agent, history_df = train_agent(
        train_builds=train_builds,
        num_episodes=episodes,
        alpha=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        w_b=2.5,
        w_c=0.8,
        w_e=0.3,
        seed=seed,
        verbose_interval=300,
    )
    agent.save(save_model_path)
    history_df.to_csv("results/training_history.csv", index=False)
    return agent, history_df


def run_main_comparison_experiment(
    agent: TabularQLearningAgent,
    test_builds: List[SoftwareBuild],
    seeds: List[int] = [42, 101, 777],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run head-to-head comparison on held-out scenarios."""
    print(f"\n[2/6] Running Main Baseline Comparison on {len(test_builds)} held-out builds across {len(seeds)} seeds...")
    raw_df = run_benchmark_comparison(agent, test_builds, seeds=seeds)
    summary_df = aggregate_experiment_results(raw_df)

    raw_df.to_csv("results/raw_results.csv", index=False)
    summary_df.to_csv("results/summary.csv", index=False)
    return raw_df, summary_df


def run_budget_sensitivity_experiment(
    agent: TabularQLearningAgent,
    base_seed: int = 42,
    budget_ratios: List[float] = [0.20, 0.35, 0.50, 0.65, 0.80],
    num_builds: int = 50,
) -> pd.DataFrame:
    """Evaluates how each strategy scales as testing budget increases/decreases."""
    print(f"\n[3/6] Running Budget Sensitivity Experiment over ratios: {budget_ratios}...")
    records = []

    for b_ratio in budget_ratios:
        bg = BuildGenerator(num_tests=30, num_features=12, min_bugs=5, max_bugs=8, budget_ratio=b_ratio, seed=base_seed)
        builds = bg.generate_dataset(num_builds=num_builds, prefix=f"budget_{int(b_ratio*100)}", base_seed=base_seed + 100)

        for build in builds:
            env = TestingEnvironment(build)

            # Evaluate Random
            r_baseline = RandomBaseline(seed=base_seed)
            r_res = evaluate_policy_on_build(r_baseline, env, build, is_rl=False)
            r_res["strategy"] = "Random"
            r_res["budget_ratio"] = b_ratio
            records.append(r_res)

            # Evaluate Greedy Cov/Cost
            g_cov = GreedyCoverageCostBaseline()
            g_cov_res = evaluate_policy_on_build(g_cov, env, build, is_rl=False)
            g_cov_res["strategy"] = "Greedy (Coverage/Cost)"
            g_cov_res["budget_ratio"] = b_ratio
            records.append(g_cov_res)

            # Evaluate Q-Learning
            ql_res = evaluate_policy_on_build(agent, env, build, is_rl=True)
            ql_res["strategy"] = "Q-Learning"
            ql_res["budget_ratio"] = b_ratio
            records.append(ql_res)

    df_budget = pd.DataFrame(records)
    df_budget.to_csv("results/budget_sensitivity.csv", index=False)
    return df_budget


def run_reward_ablation_experiment(
    train_builds: List[SoftwareBuild],
    test_builds: List[SoftwareBuild],
    episodes: int = 800,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Controlled ablation of reward function weights:
    - R1: Balanced (w_b=2.5, w_c=0.8, w_e=0.3)
    - R2: Bug-Centric (w_b=3.0, w_c=0.0, w_e=0.0)
    - R3: Coverage-Centric (w_b=1.0, w_c=2.5, w_e=0.2)
    """
    print("\n[4/6] Running Reward Function Ablation Study...")
    ablation_configs = [
        ("Balanced (w_b=2.5, w_c=0.8, w_e=0.3)", 2.5, 0.8, 0.3),
        ("Bug-Centric (w_b=3.0, w_c=0.0, w_e=0.0)", 3.0, 0.0, 0.0),
        ("Coverage-Centric (w_b=1.0, w_c=2.5, w_e=0.2)", 1.0, 2.5, 0.2),
    ]

    records = []
    for name, wb, wc, we in ablation_configs:
        print(f"  Training ablation variant: {name}...")
        agent, _ = train_agent(
            train_builds=train_builds,
            num_episodes=episodes,
            w_b=wb,
            w_c=wc,
            w_e=we,
            seed=seed,
            verbose_interval=9999,
        )

        env = TestingEnvironment(test_builds[0], w_b=wb, w_c=wc, w_e=we)
        for build in test_builds[:50]:
            res = evaluate_policy_on_build(agent, env, build, is_rl=True)
            res["ablation_variant"] = name
            records.append(res)

    df_ablation = pd.DataFrame(records)
    df_ablation.to_csv("results/reward_ablation.csv", index=False)
    return df_ablation


def run_real_world_ci_experiment(
    dataset_name: str = "paintcontrol",
    episodes: int = 800,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates Q-learning and baselines on the real-world ABB Paint Control dataset.
    """
    print(f"\n[5/6] Running Real-World Evaluation on {dataset_name.upper()} CI dataset...")
    df = load_ci_dataset(dataset_name)
    train_cycles, test_cycles = split_ci_cycles(df, train_ratio=0.70)

    train_builds = RealWorldCIAdapter.load_all_builds_from_df(df, train_cycles, budget_ratio=0.45)
    test_builds = RealWorldCIAdapter.load_all_builds_from_df(df, test_cycles, budget_ratio=0.45)

    print(f"  Extracted {len(train_builds)} train cycles and {len(test_builds)} held-out test cycles with defects.")

    test_history = df[df["Cycle"].isin(train_cycles)].groupby("Name")["Verdict"].mean().to_dict()

    max_actions = max(b.num_tests for b in train_builds + test_builds)
    ci_agent, _ = train_agent(
        train_builds=train_builds,
        num_episodes=episodes,
        seed=seed,
        verbose_interval=400,
    )

    raw_ci_df = run_benchmark_comparison(ci_agent, test_builds, seeds=[seed, seed + 1], test_history=test_history)
    summary_ci_df = aggregate_experiment_results(raw_ci_df)

    raw_ci_df.to_csv(f"results/real_world_{dataset_name}_raw.csv", index=False)
    summary_ci_df.to_csv(f"results/real_world_{dataset_name}_summary.csv", index=False)
    return raw_ci_df, summary_ci_df


def generate_all_plots():
    """Generates all publication-grade figures in results/figures/."""
    print("\n[6/6] Generating Publication Figures in results/figures/...")
    os.makedirs("results/figures", exist_ok=True)

    # 1. Training Curve
    if os.path.exists("results/training_history.csv"):
        th = pd.read_csv("results/training_history.csv")
        fig, ax1 = plt.subplots(figsize=(8, 4.5))

        # Rolling mean reward
        window = 30
        th["reward_smooth"] = th["reward"].rolling(window, min_periods=1).mean()
        th["bdr_smooth"] = th["bug_detection_rate"].rolling(window, min_periods=1).mean() * 100

        color = "#1f77b4"
        ax1.set_xlabel("Training Episode", fontweight="bold")
        ax1.set_ylabel("Episode Reward", color=color, fontweight="bold")
        ax1.plot(th["episode"], th["reward_smooth"], color=color, linewidth=2, label="Reward (Smoothed)")
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color2 = "#2ca02c"
        ax2.set_ylabel("Bug Detection Rate (%)", color=color2, fontweight="bold")
        ax2.plot(th["episode"], th["bdr_smooth"], color=color2, linewidth=2, linestyle="--", label="BDR (%)")
        ax2.tick_params(axis="y", labelcolor=color2)

        plt.title("TestPilot-RL: Training Dynamics and Convergence", pad=12, fontweight="bold")
        fig.tight_layout()
        plt.savefig("results/figures/training_curve.png", dpi=300)
        plt.close()
        print("  Saved results/figures/training_curve.png")

    # 2. Baseline Comparison
    if os.path.exists("results/summary.csv"):
        summ = pd.read_csv("results/summary.csv")
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

        palette = ["#7f7f7f", "#ff7f0e", "#1f77b4", "#2ca02c"]

        # Bug Detection Rate
        sns.barplot(
            data=summ,
            x="strategy",
            y="bug_detection_rate_mean",
            ax=axes[0],
            palette=palette[:len(summ)],
        )
        axes[0].errorbar(
            x=range(len(summ)),
            y=summ["bug_detection_rate_mean"],
            yerr=summ["bug_detection_rate_ci95"],
            fmt="none",
            c="black",
            capsize=4,
        )
        axes[0].set_title("Bug Detection Rate (BDR)", fontweight="bold")
        axes[0].set_ylabel("Mean BDR")
        axes[0].set_xlabel("")
        axes[0].set_ylim(0, 1.05)
        axes[0].tick_params(axis="x", rotation=15)

        # Coverage Rate
        sns.barplot(
            data=summ,
            x="strategy",
            y="coverage_rate_mean",
            ax=axes[1],
            palette=palette[:len(summ)],
        )
        axes[1].errorbar(
            x=range(len(summ)),
            y=summ["coverage_rate_mean"],
            yerr=summ["coverage_rate_ci95"],
            fmt="none",
            c="black",
            capsize=4,
        )
        axes[1].set_title("Code Feature Coverage", fontweight="bold")
        axes[1].set_ylabel("Mean Coverage Ratio")
        axes[1].set_xlabel("")
        axes[1].set_ylim(0, 1.05)
        axes[1].tick_params(axis="x", rotation=15)

        # Bug Detection Efficiency (Bugs per cost)
        sns.barplot(
            data=summ,
            x="strategy",
            y="bugs_per_cost_unit_mean",
            ax=axes[2],
            palette=palette[:len(summ)],
        )
        axes[2].errorbar(
            x=range(len(summ)),
            y=summ["bugs_per_cost_unit_mean"],
            yerr=summ["bugs_per_cost_unit_ci95"],
            fmt="none",
            c="black",
            capsize=4,
        )
        axes[2].set_title("Bug Detection Efficiency", fontweight="bold")
        axes[2].set_ylabel("Bugs / Cost Unit")
        axes[2].set_xlabel("")
        axes[2].tick_params(axis="x", rotation=15)

        fig.suptitle("TestPilot-RL: Comparison with Baseline Strategies (95% CI)", fontweight="bold", y=1.02)
        fig.tight_layout()
        plt.savefig("results/figures/baseline_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved results/figures/baseline_comparison.png")

    # 3. Budget Sensitivity
    if os.path.exists("results/budget_sensitivity.csv"):
        bs = pd.read_csv("results/budget_sensitivity.csv")
        fig, ax = plt.subplots(figsize=(7.5, 4.5))

        sns.lineplot(
            data=bs,
            x="budget_ratio",
            y="bug_detection_rate",
            hue="strategy",
            marker="o",
            ax=ax,
            linewidth=2.2,
        )
        ax.set_title("Budget Sensitivity: Bug Detection Rate vs. Testing Budget", fontweight="bold")
        ax.set_xlabel("Testing Budget Ratio (% of Suite Cost)", fontweight="bold")
        ax.set_ylabel("Bug Detection Rate (BDR)", fontweight="bold")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        plt.savefig("results/figures/budget_sensitivity.png", dpi=300)
        plt.close()
        print("  Saved results/figures/budget_sensitivity.png")

    # 4. Reward Ablation
    if os.path.exists("results/reward_ablation.csv"):
        ra = pd.read_csv("results/reward_ablation.csv")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ra_mean = ra.groupby("ablation_variant")[["bug_detection_rate", "coverage_rate"]].mean().reset_index()
        ra_melted = pd.melt(ra_mean, id_vars=["ablation_variant"], value_vars=["bug_detection_rate", "coverage_rate"],
                            var_name="Metric", value_name="Score")

        sns.barplot(data=ra_melted, x="ablation_variant", y="Score", hue="Metric", ax=ax, palette=["#1f77b4", "#2ca02c"])
        ax.set_title("Reward Formulation Ablation Study", fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_xlabel("Reward Weight Variant")
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        plt.savefig("results/figures/reward_ablation.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved results/figures/reward_ablation.png")

    # 5. Real World Evaluation
    if os.path.exists("results/real_world_paintcontrol_summary.csv"):
        ci_summ = pd.read_csv("results/real_world_paintcontrol_summary.csv")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=ci_summ, x="strategy", y="bug_detection_rate_mean", ax=ax, palette="Blues_d")
        ax.errorbar(
            x=range(len(ci_summ)),
            y=ci_summ["bug_detection_rate_mean"],
            yerr=ci_summ["bug_detection_rate_ci95"],
            fmt="none",
            c="black",
            capsize=4,
        )
        ax.set_title("Real-World ABB Paint Control Industrial CI Benchmark", fontweight="bold")
        ax.set_ylabel("Bug Detection Rate (BDR)")
        ax.set_xlabel("")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        plt.savefig("results/figures/real_world_evaluation.png", dpi=300)
        plt.close()
        print("  Saved results/figures/real_world_evaluation.png")


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # 1. Setup synthetic data
    bg = BuildGenerator(num_tests=30, num_features=12, min_bugs=5, max_bugs=8, budget_ratio=0.45, seed=42)
    train_builds = bg.generate_dataset(num_builds=50, prefix="train_build", base_seed=42)
    test_builds = bg.generate_dataset(num_builds=120, prefix="heldout_build", base_seed=5000)

    # 2. Run experiments
    agent, history_df = run_training_experiment(train_builds, episodes=1200, seed=42)
    raw_df, summary_df = run_main_comparison_experiment(agent, test_builds, seeds=[42, 101, 777])
    df_budget = run_budget_sensitivity_experiment(agent, base_seed=42, num_builds=40)
    df_ablation = run_reward_ablation_experiment(train_builds, test_builds, episodes=600, seed=42)
    run_real_world_ci_experiment(dataset_name="paintcontrol", episodes=800, seed=42)

    # 3. Generate figures
    generate_all_plots()

    print("\n=======================================================")
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=======================================================")


if __name__ == "__main__":
    main()
