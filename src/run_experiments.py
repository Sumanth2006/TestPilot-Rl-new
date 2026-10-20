"""
Master scientific experiment runner for TestPilot-RL.

Executes all experimental suites:
1. Training and convergence analysis (Training Curve)
2. Main comparison against Random and Greedy baselines (120 held-out scenarios, 3 seeds)
3. Budget sensitivity analysis (20%, 35%, 50%, 65%, 80%)
4. Reward weight ablation (Balanced vs. Pure Bug vs. Coverage Heavy)
5. Ablation: Linear Q-Learning vs. Tabular Q-Learning
6. Real-world CI evaluation: ABB Paint Control AND IOF/ROL datasets
7. Generates publication-ready figures in results/figures/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional

from src.simulator import BuildGenerator, RealWorldCIAdapter, SoftwareBuild
from src.environment import TestingEnvironment
from src.q_learning import LinearQLearningAgent, TabularQLearningAgent
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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_synthetic_test_history(train_builds: List[SoftwareBuild]) -> Dict[int, float]:
    """Computes empirical per-test failure rate from a list of training builds."""
    fail_counts: Dict[int, int] = {}
    exec_counts: Dict[int, int] = {}
    for b in train_builds:
        for t in b.test_cases:
            exec_counts[t] = exec_counts.get(t, 0) + 1
            if len(b.bug_detection_map.get(t, set())) > 0:
                fail_counts[t] = fail_counts.get(t, 0) + 1
    return {t: fail_counts.get(t, 0) / max(1, exec_counts[t]) for t in exec_counts}


def _build_ci_test_history(df: pd.DataFrame, train_cycles: List[int]) -> Dict[int, float]:
    """
    Computes per-test failure rates from CI log data for the training cycles.
    Keys are integer test indices as used by RealWorldCIAdapter.
    """
    train_df = df[df["Cycle"].isin(train_cycles)]
    # Name-level failure rate
    name_fail_rate = train_df.groupby("Name")["Verdict"].mean().to_dict()

    # Map name -> integer index based on the first cycle's ordering
    # (same ordering used by RealWorldCIAdapter.from_ci_dataframe)
    all_names = df["Name"].unique().tolist()
    name_to_idx = {name: idx for idx, name in enumerate(all_names)}
    return {name_to_idx[n]: v for n, v in name_fail_rate.items() if n in name_to_idx}


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 1: Training
# ──────────────────────────────────────────────────────────────────────────────

def run_training_experiment(
    train_builds: List[SoftwareBuild],
    episodes: int = 2000,
    seed: int = 42,
    test_history: Optional[Dict[int, float]] = None,
    save_model_path: str = "models/q_table.npy",
) -> Tuple[LinearQLearningAgent, pd.DataFrame]:
    """Train primary Linear Q-learning agent and record trajectory."""
    print("\n[1/7] Running Training Experiment (Linear Q-Learning)...")
    agent, history_df = train_agent(
        train_builds=train_builds,
        num_episodes=episodes,
        alpha=0.008,
        gamma=0.97,
        epsilon_start=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.997,
        w_b=3.0,
        w_c=0.8,
        w_e=0.2,
        test_history=test_history,
        shaping_coef=2.5,
        replay_capacity=30000,
        batch_size=64,
        seed=seed,
        use_linear=True,
        verbose_interval=400,
    )
    agent.save(save_model_path)
    history_df.to_csv("results/training_history.csv", index=False)
    return agent, history_df


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 2: Main Comparison
# ──────────────────────────────────────────────────────────────────────────────

def run_main_comparison_experiment(
    agent: LinearQLearningAgent,
    test_builds: List[SoftwareBuild],
    seeds: List[int] = [42, 101, 777],
    test_history: Optional[Dict[int, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run head-to-head comparison on held-out scenarios."""
    print(f"\n[2/7] Running Main Baseline Comparison on {len(test_builds)} held-out builds across {len(seeds)} seeds...")
    raw_df = run_benchmark_comparison(agent, test_builds, seeds=seeds, test_history=test_history)
    summary_df = aggregate_experiment_results(raw_df)

    raw_df.to_csv("results/raw_results.csv", index=False)
    summary_df.to_csv("results/summary.csv", index=False)
    return raw_df, summary_df


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 3: Budget Sensitivity
# ──────────────────────────────────────────────────────────────────────────────

def run_budget_sensitivity_experiment(
    agent: LinearQLearningAgent,
    base_seed: int = 42,
    budget_ratios: List[float] = [0.20, 0.35, 0.50, 0.65, 0.80],
    num_builds: int = 60,
    test_history: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """Evaluates how each strategy scales as testing budget increases/decreases."""
    print(f"\n[3/7] Running Budget Sensitivity Experiment over ratios: {budget_ratios}...")
    records = []

    for b_ratio in budget_ratios:
        bg = BuildGenerator(num_tests=30, num_features=12, min_bugs=5, max_bugs=8, budget_ratio=b_ratio, seed=base_seed)
        builds = bg.generate_dataset(num_builds=num_builds, prefix=f"budget_{int(b_ratio*100)}", base_seed=base_seed + 100)
        th = test_history or _build_synthetic_test_history(builds[:30])

        for build in builds:
            env = TestingEnvironment(build)

            r_baseline = RandomBaseline(seed=base_seed)
            r_res = evaluate_policy_on_build(r_baseline, env, build, is_rl=False)
            r_res["strategy"] = "Random"
            r_res["budget_ratio"] = b_ratio
            records.append(r_res)

            g_cov = GreedyCoverageCostBaseline()
            g_cov_res = evaluate_policy_on_build(g_cov, env, build, is_rl=False)
            g_cov_res["strategy"] = "Greedy (Coverage/Cost)"
            g_cov_res["budget_ratio"] = b_ratio
            records.append(g_cov_res)

            g_fail = GreedyFailureRateBaseline(test_history=th)
            g_fail_res = evaluate_policy_on_build(g_fail, env, build, is_rl=False)
            g_fail_res["strategy"] = "Greedy (Failure/Cost)"
            g_fail_res["budget_ratio"] = b_ratio
            records.append(g_fail_res)

            ql_res = evaluate_policy_on_build(agent, env, build, is_rl=True)
            ql_res["strategy"] = "Q-Learning (Linear)"
            ql_res["budget_ratio"] = b_ratio
            records.append(ql_res)

    df_budget = pd.DataFrame(records)
    df_budget.to_csv("results/budget_sensitivity.csv", index=False)
    return df_budget


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 4: Reward Ablation
# ──────────────────────────────────────────────────────────────────────────────

def run_reward_ablation_experiment(
    train_builds: List[SoftwareBuild],
    test_builds: List[SoftwareBuild],
    episodes: int = 1200,
    seed: int = 42,
    test_history: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """
    Controlled ablation of reward function weights:
    - R1: Balanced (w_b=3.0, w_c=0.8, w_e=0.2)
    - R2: Bug-Centric (w_b=5.0, w_c=0.0, w_e=0.0)
    - R3: Coverage-Centric (w_b=1.0, w_c=3.0, w_e=0.2)
    """
    print("\n[4/7] Running Reward Function Ablation Study...")
    ablation_configs = [
        ("Balanced (w_b=3.0, w_c=0.8, w_e=0.2)", 3.0, 0.8, 0.2),
        ("Bug-Centric (w_b=5.0, w_c=0.0, w_e=0.0)", 5.0, 0.0, 0.0),
        ("Coverage-Centric (w_b=1.0, w_c=3.0, w_e=0.2)", 1.0, 3.0, 0.2),
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
            test_history=test_history,
            seed=seed,
            use_linear=True,
            verbose_interval=9999,
        )

        env = TestingEnvironment(test_builds[0], w_b=wb, w_c=wc, w_e=we)
        for build in test_builds[:60]:
            res = evaluate_policy_on_build(agent, env, build, is_rl=True)
            res["ablation_variant"] = name
            records.append(res)

    df_ablation = pd.DataFrame(records)
    df_ablation.to_csv("results/reward_ablation.csv", index=False)
    return df_ablation


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 5: Linear vs. Tabular Ablation
# ──────────────────────────────────────────────────────────────────────────────

def run_agent_ablation_experiment(
    train_builds: List[SoftwareBuild],
    test_builds: List[SoftwareBuild],
    episodes: int = 1200,
    seed: int = 42,
    test_history: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """
    Compares Linear Q-Learning vs. Legacy Tabular Q-Learning to demonstrate
    the improvement from function approximation.
    """
    print("\n[5/7] Running Agent Ablation: Linear vs Tabular Q-Learning...")
    records = []

    for use_linear, label in [(True, "Linear Q-Learning (ours)"), (False, "Tabular Q-Learning (legacy)")]:
        print(f"  Training {label}...")
        agent, _ = train_agent(
            train_builds=train_builds,
            num_episodes=episodes,
            test_history=test_history,
            seed=seed,
            use_linear=use_linear,
            verbose_interval=9999,
        )
        env = TestingEnvironment(test_builds[0])
        for build in test_builds[:60]:
            res = evaluate_policy_on_build(agent, env, build, is_rl=True)
            res["agent_variant"] = label
            records.append(res)

    df = pd.DataFrame(records)
    df.to_csv("results/agent_ablation.csv", index=False)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 6: Real-World CI Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_real_world_ci_experiment(
    dataset_name: str = "paintcontrol",
    episodes: int = 1500,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates Q-learning and baselines on the real-world ABB CI datasets.
    Uses a proper train/test split and computes per-test historical failure rates
    from training cycles only (no leakage).
    """
    print(f"\n[6/7] Running Real-World Evaluation on {dataset_name.upper()} CI dataset...")
    df = load_ci_dataset(dataset_name)
    train_cycles, test_cycles = split_ci_cycles(df, train_ratio=0.70)

    train_builds = RealWorldCIAdapter.load_all_builds_from_df(df, train_cycles, budget_ratio=0.45)
    test_builds = RealWorldCIAdapter.load_all_builds_from_df(df, test_cycles, budget_ratio=0.45)

    print(f"  Extracted {len(train_builds)} train cycles and {len(test_builds)} held-out test cycles with defects.")

    # Compute test_history with consistent integer indexing
    test_history = _build_ci_test_history(df, train_cycles)

    ci_agent, _ = train_agent(
        train_builds=train_builds,
        num_episodes=episodes,
        test_history=test_history,
        seed=seed,
        use_linear=True,
        shaping_coef=3.0,  # stronger shaping on real data (sparser signals)
        alpha=0.006,
        gamma=0.97,
        epsilon_min=0.02,
        verbose_interval=500,
    )

    raw_ci_df = run_benchmark_comparison(ci_agent, test_builds, seeds=[seed, seed + 1, seed + 2], test_history=test_history)
    summary_ci_df = aggregate_experiment_results(raw_ci_df)

    raw_ci_df.to_csv(f"results/real_world_{dataset_name}_raw.csv", index=False)
    summary_ci_df.to_csv(f"results/real_world_{dataset_name}_summary.csv", index=False)

    print(f"\n  {dataset_name.upper()} Summary:")
    for _, row in summary_ci_df.iterrows():
        print(f"    {row['strategy']:30s} BDR={row['bug_detection_rate_mean']:.3f}  APFD={row['apfd_mean']:.3f}")

    return raw_ci_df, summary_ci_df


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 7: Plot Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_plots():
    """Generates all publication-grade figures in results/figures/."""
    print("\n[7/7] Generating Publication Figures in results/figures/...")
    os.makedirs("results/figures", exist_ok=True)

    PALETTE = {
        "Random": "#7f7f7f",
        "Greedy (Coverage/Cost)": "#ff7f0e",
        "Greedy (Failure/Cost)": "#d62728",
        "Q-Learning": "#2ca02c",
        "Q-Learning (Linear)": "#2ca02c",
        "Tabular Q-Learning (legacy)": "#9467bd",
        "Linear Q-Learning (ours)": "#2ca02c",
    }

    # 1. Training Curve
    if os.path.exists("results/training_history.csv"):
        th = pd.read_csv("results/training_history.csv")
        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        window = 50
        th["reward_smooth"] = th["reward"].rolling(window, min_periods=1).mean()
        th["bdr_smooth"] = th["bug_detection_rate"].rolling(window, min_periods=1).mean() * 100

        color = "#1f77b4"
        ax1.set_xlabel("Training Episode", fontweight="bold")
        ax1.set_ylabel("Episode Reward", color=color, fontweight="bold")
        ax1.plot(th["episode"], th["reward_smooth"], color=color, linewidth=2, label="Reward (Smoothed)")
        ax1.fill_between(
            th["episode"],
            th["reward"].rolling(window, min_periods=1).quantile(0.25),
            th["reward"].rolling(window, min_periods=1).quantile(0.75),
            alpha=0.15, color=color,
        )
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color2 = "#2ca02c"
        ax2.set_ylabel("Bug Detection Rate (%)", color=color2, fontweight="bold")
        ax2.plot(th["episode"], th["bdr_smooth"], color=color2, linewidth=2, linestyle="--", label="BDR (%)")
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(0, 105)

        plt.title("TestPilot-RL: Training Dynamics and Convergence (Linear Q-Learning)", pad=12, fontweight="bold")
        fig.tight_layout()
        plt.savefig("results/figures/training_curve.png", dpi=300)
        plt.close()
        print("  Saved results/figures/training_curve.png")

    # 2. Baseline Comparison (synthetic)
    if os.path.exists("results/summary.csv"):
        summ = pd.read_csv("results/summary.csv")
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        colors = [PALETTE.get(s, "#1f77b4") for s in summ["strategy"]]
        metrics = [
            ("bug_detection_rate_mean", "bug_detection_rate_ci95", "Bug Detection Rate (BDR)", "Mean BDR", (0, 1.05)),
            ("coverage_rate_mean", "coverage_rate_ci95", "Code Feature Coverage", "Mean Coverage Ratio", (0, 1.05)),
            ("apfd_mean", "apfd_ci95", "APFD (Avg % Faults Detected)", "APFD Score", (0, 1.05)),
        ]

        for ax, (col_mean, col_ci, title, ylabel, ylim) in zip(axes, metrics):
            bars = ax.bar(summ["strategy"], summ[col_mean], color=colors, edgecolor="white", linewidth=0.8)
            ax.errorbar(
                x=range(len(summ)),
                y=summ[col_mean],
                yerr=summ[col_ci],
                fmt="none",
                c="black",
                capsize=5,
                linewidth=1.5,
            )
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("")
            ax.set_ylim(*ylim)
            ax.tick_params(axis="x", rotation=20)
            # Highlight Q-Learning bar with a border
            for bar, strat in zip(bars, summ["strategy"]):
                if "Q-Learning" in str(strat):
                    bar.set_edgecolor("#000000")
                    bar.set_linewidth(2)

        fig.suptitle("TestPilot-RL: Synthetic Benchmark Comparison vs. Baselines (95% CI)", fontweight="bold", y=1.02)
        fig.tight_layout()
        plt.savefig("results/figures/baseline_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved results/figures/baseline_comparison.png")

    # 3. Budget Sensitivity
    if os.path.exists("results/budget_sensitivity.csv"):
        bs = pd.read_csv("results/budget_sensitivity.csv")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        strat_colors = {s: PALETTE.get(s, "#1f77b4") for s in bs["strategy"].unique()}

        for ax, (metric, ylabel) in zip(axes, [
            ("bug_detection_rate", "Bug Detection Rate (BDR)"),
            ("apfd", "APFD Score"),
        ]):
            for strat, grp in bs.groupby("strategy"):
                mean = grp.groupby("budget_ratio")[metric].mean()
                std = grp.groupby("budget_ratio")[metric].std()
                lw = 2.5 if "Q-Learning" in strat else 1.8
                ls = "-" if "Q-Learning" in strat else "--"
                ax.plot(mean.index, mean.values, marker="o", label=strat,
                        color=strat_colors.get(strat, "#888"),
                        linewidth=lw, linestyle=ls)
                ax.fill_between(mean.index, mean - std, mean + std, alpha=0.12,
                                color=strat_colors.get(strat, "#888"))

            ax.set_xlabel("Testing Budget Ratio (fraction of full suite cost)", fontweight="bold")
            ax.set_ylabel(ylabel, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=9)

        axes[0].set_title("Budget Sensitivity: Bug Detection Rate", fontweight="bold")
        axes[1].set_title("Budget Sensitivity: APFD Score", fontweight="bold")
        fig.suptitle("TestPilot-RL: Performance vs. Testing Budget", fontweight="bold")
        fig.tight_layout()
        plt.savefig("results/figures/budget_sensitivity.png", dpi=300)
        plt.close()
        print("  Saved results/figures/budget_sensitivity.png")

    # 4. Reward Ablation
    if os.path.exists("results/reward_ablation.csv"):
        ra = pd.read_csv("results/reward_ablation.csv")
        fig, ax = plt.subplots(figsize=(9, 5))
        ra_mean = ra.groupby("ablation_variant")[["bug_detection_rate", "coverage_rate", "apfd"]].mean().reset_index()
        ra_melted = pd.melt(ra_mean, id_vars=["ablation_variant"],
                            value_vars=["bug_detection_rate", "coverage_rate", "apfd"],
                            var_name="Metric", value_name="Score")
        metric_labels = {"bug_detection_rate": "BDR", "coverage_rate": "Coverage", "apfd": "APFD"}
        ra_melted["Metric"] = ra_melted["Metric"].map(metric_labels)

        sns.barplot(data=ra_melted, x="ablation_variant", y="Score", hue="Metric",
                    ax=ax, palette=["#1f77b4", "#2ca02c", "#ff7f0e"])
        ax.set_title("Reward Formulation Ablation Study", fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_xlabel("Reward Weight Variant")
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Metric")
        fig.tight_layout()
        plt.savefig("results/figures/reward_ablation.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved results/figures/reward_ablation.png")

    # 5. Agent Ablation (Linear vs Tabular)
    if os.path.exists("results/agent_ablation.csv"):
        aa = pd.read_csv("results/agent_ablation.csv")
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        aa_mean = aa.groupby("agent_variant")[["bug_detection_rate", "coverage_rate", "apfd"]].mean().reset_index()

        for ax, col, title in zip(
            axes,
            ["bug_detection_rate", "coverage_rate", "apfd"],
            ["Bug Detection Rate", "Coverage Rate", "APFD"],
        ):
            colors = ["#2ca02c" if "Linear" in v else "#9467bd" for v in aa_mean["agent_variant"]]
            bars = ax.bar(aa_mean["agent_variant"], aa_mean[col], color=colors, edgecolor="white")
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.1)
            ax.tick_params(axis="x", rotation=15)

        fig.suptitle("Agent Ablation: Linear Q-Learning vs. Tabular Q-Learning", fontweight="bold")
        fig.tight_layout()
        plt.savefig("results/figures/agent_ablation.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved results/figures/agent_ablation.png")

    # 6. Real-World Evaluation — both datasets
    for dataset, title_suffix in [
        ("paintcontrol", "ABB Paint Control"),
        ("iofrol", "ABB IOF/ROL"),
    ]:
        path = f"results/real_world_{dataset}_summary.csv"
        if os.path.exists(path):
            ci_summ = pd.read_csv(path)
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            colors = [PALETTE.get(s, "#1f77b4") for s in ci_summ["strategy"]]

            for ax, (col_mean, col_ci, ylabel) in zip(axes, [
                ("bug_detection_rate_mean", "bug_detection_rate_ci95", "Bug Detection Rate (BDR)"),
                ("apfd_mean", "apfd_ci95", "APFD Score"),
            ]):
                bars = ax.bar(ci_summ["strategy"], ci_summ[col_mean], color=colors, edgecolor="white")
                ax.errorbar(
                    x=range(len(ci_summ)),
                    y=ci_summ[col_mean],
                    yerr=ci_summ[col_ci],
                    fmt="none",
                    c="black",
                    capsize=5,
                    linewidth=1.5,
                )
                ax.set_ylabel(ylabel, fontweight="bold")
                ax.set_ylim(0, 1.05)
                ax.tick_params(axis="x", rotation=20)
                for bar, strat in zip(bars, ci_summ["strategy"]):
                    if "Q-Learning" in str(strat):
                        bar.set_edgecolor("#000000")
                        bar.set_linewidth(2)

            axes[0].set_title("Bug Detection Rate (BDR)", fontweight="bold")
            axes[1].set_title("APFD Score", fontweight="bold")
            fig.suptitle(f"Real-World {title_suffix} Industrial CI Benchmark", fontweight="bold")
            fig.tight_layout()
            plt.savefig(f"results/figures/real_world_{dataset}.png", dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  Saved results/figures/real_world_{dataset}.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # ── Synthetic dataset setup ──
    print("Setting up synthetic training and test builds...")
    bg_train = BuildGenerator(num_tests=30, num_features=12, min_bugs=5, max_bugs=8, budget_ratio=0.45, seed=42)
    bg_test = BuildGenerator(num_tests=30, num_features=12, min_bugs=5, max_bugs=8, budget_ratio=0.45, seed=99)
    train_builds = bg_train.generate_dataset(num_builds=80, prefix="train_build", base_seed=42)
    test_builds = bg_test.generate_dataset(num_builds=120, prefix="heldout_build", base_seed=5000)

    syn_hist = _build_synthetic_test_history(train_builds)

    # ── Run experiments ──
    agent, history_df = run_training_experiment(train_builds, episodes=2000, test_history=syn_hist, seed=42)
    raw_df, summary_df = run_main_comparison_experiment(agent, test_builds, seeds=[42, 101, 777], test_history=syn_hist)

    print("\n===== SYNTHETIC BENCHMARK SUMMARY =====")
    for _, row in summary_df.iterrows():
        print(f"  {row['strategy']:35s} BDR={row['bug_detection_rate_mean']:.3f} ± {row['bug_detection_rate_std']:.3f}  "
              f"APFD={row['apfd_mean']:.3f}")

    df_budget = run_budget_sensitivity_experiment(agent, base_seed=42, num_builds=50, test_history=syn_hist)
    df_ablation = run_reward_ablation_experiment(train_builds, test_builds, episodes=1200, seed=42, test_history=syn_hist)
    df_agent_ablation = run_agent_ablation_experiment(train_builds, test_builds, episodes=1200, seed=42, test_history=syn_hist)

    # Real-world experiments
    run_real_world_ci_experiment(dataset_name="paintcontrol", episodes=1500, seed=42)
    run_real_world_ci_experiment(dataset_name="iofrol", episodes=1500, seed=42)

    # Generate plots
    generate_all_plots()

    print("\n=======================================================")
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=======================================================")
    print("\nKey result files:")
    print("  results/summary.csv              — synthetic benchmark comparison")
    print("  results/real_world_paintcontrol_summary.csv — PaintControl CI results")
    print("  results/real_world_iofrol_summary.csv       — IOF/ROL CI results")
    print("  results/figures/                 — all publication figures")


if __name__ == "__main__":
    main()
