"""
Master scientific experiment runner for TestPilot-RL.

Executes all experimental suites (with zero data leakage and multi-seed training):
1. Multi-seed training and convergence analysis (3 training seeds: 42, 101, 777)
2. Main comparison against Random, Greedy Coverage/Cost, Greedy Failure/Cost, and symmetric Greedy Recency/Cost
3. Budget sensitivity analysis (20%, 35%, 50%, 65%, 80%) using strictly separate training builds per budget
4. Reward weight ablation (Balanced vs. Bug-Only vs. Coverage-Only vs. Cost-Penalized)
5. State feature ablation (Full 12-Feature State vs. No-Recency State vs. No-Coverage State)
6. Agent Architecture Ablation: Dueling DQN vs. Linear Q-Learning vs. Contextual Bandit vs. Tabular Q-Learning
7. Real-world CI evaluation: ABB Paint Control AND IOF/ROL datasets (reporting total vs. defect cycles)
8. Generates publication-ready figures in results/figures/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional

from src.simulator import BuildGenerator, RealWorldCIAdapter, SoftwareBuild
from src.environment import TestingEnvironment
from src.q_learning import (
    DuelingDQNAgent,
    LinearQLearningAgent,
    ContextualBanditAgent,
    TabularQLearningAgent,
    NUM_FEATURES,
)
from src.baselines import (
    RandomBaseline,
    GreedyCoverageCostBaseline,
    GreedyFailureRateBaseline,
    GreedyRecencyCostBaseline,
)
from src.train import train_agent, build_canonical_ci_history
from src.evaluate import evaluate_policy_on_build, run_benchmark_comparison
from src.metrics import (
    aggregate_experiment_results,
    compute_apfd,
    compute_napfd,
    compute_paired_cycle_comparisons,
)
from src.data_loader import load_ci_dataset, split_ci_cycles


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

def _build_synthetic_test_history(train_builds: List[SoftwareBuild]) -> Dict[Any, float]:
    """Computes empirical per-test failure rate strictly from training builds."""
    fail_counts: Dict[int, int] = {}
    exec_counts: Dict[int, int] = {}
    for b in train_builds:
        for t in b.test_cases:
            exec_counts[t] = exec_counts.get(t, 0) + 1
            if len(b.bug_detection_map.get(t, set())) > 0:
                fail_counts[t] = fail_counts.get(t, 0) + 1

    history: Dict[Any, float] = {}
    for t in exec_counts:
        rate = fail_counts.get(t, 0) / max(1, exec_counts[t])
        history[t] = rate
        history[f"SynTest_{t:02d}"] = rate
    return history


def _build_ci_test_history(df: pd.DataFrame, train_cycles: List[int]) -> Dict[Any, float]:
    """Computes per-test failure rates from CI log data strictly for the training cycles."""
    return build_canonical_ci_history(df, train_cycles)


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 1: Multi-Seed Training & Convergence
# ──────────────────────────────────────────────────────────────────────────────

def run_training_experiment(
    train_builds: List[SoftwareBuild],
    episodes: int = 450,
    seeds: List[int] = [42, 101, 777],
    test_history: Optional[Dict[Any, float]] = None,
    save_model_path: str = "models/q_table.npy",
) -> Tuple[LinearQLearningAgent, DuelingDQNAgent, Dict[int, Dict[str, Any]], pd.DataFrame]:
    """
    Trains Linear Q-Learning and Dueling DQN across multiple independent training seeds
    from random near-zero initialization (shaping_coef=0.0).
    """
    print(f"\n[1/8] Running Multi-Seed Training Experiment across seeds {seeds}...")
    trained_by_seed: Dict[int, Dict[str, Any]] = {}
    all_history_dfs = []

    primary_linear = None
    primary_dqn = None

    for s in seeds:
        lin_ag, h_df = train_agent(
            train_builds=train_builds,
            num_episodes=episodes,
            alpha=0.015,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_min=0.02,
            epsilon_decay=0.993,
            w_b=3.0,
            w_c=0.8,
            w_e=0.2,
            test_history=test_history,
            shaping_coef=0.0,
            replay_capacity=8000,
            batch_size=32,
            seed=s,
            use_linear=True,
            agent_type="linear",
            verbose_interval=225,
        )
        h_df["train_seed"] = s
        all_history_dfs.append(h_df)

        dqn_ag, _ = train_agent(
            train_builds=train_builds,
            num_episodes=episodes,
            alpha=0.008,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_min=0.02,
            epsilon_decay=0.993,
            w_b=3.0,
            w_c=0.8,
            w_e=0.2,
            test_history=test_history,
            shaping_coef=0.0,
            replay_capacity=8000,
            batch_size=32,
            seed=s,
            agent_type="dueling_dqn",
            verbose_interval=225,
        )

        trained_by_seed[s] = {
            "Q-Learning": lin_ag,
            "Dueling DQN (Online RL)": dqn_ag,
        }
        if primary_linear is None:
            primary_linear = lin_ag
            primary_dqn = dqn_ag
            lin_ag.save(save_model_path)
            dqn_ag.save("models/dueling_dqn.npy")

    history_df = pd.concat(all_history_dfs, ignore_index=True)
    history_df.to_csv("results/training_history.csv", index=False)
    return primary_linear, primary_dqn, trained_by_seed, history_df


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 2: Main Comparison
# ──────────────────────────────────────────────────────────────────────────────

def run_main_comparison_experiment(
    agent: LinearQLearningAgent,
    dqn_agent: DuelingDQNAgent,
    test_builds: List[SoftwareBuild],
    seeds: List[int] = [42, 101, 777],
    test_history: Optional[Dict[Any, float]] = None,
    trained_agents_by_seed: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run head-to-head comparison on held-out scenarios across multiple trained seeds."""
    print(f"\n[2/8] Running Main Baseline Comparison on {len(test_builds)} held-out builds across {len(seeds)} training/eval seeds...")
    raw_df = run_benchmark_comparison(
        agent,
        test_builds,
        seeds=seeds,
        test_history=test_history,
        extra_agents=[("Dueling DQN (Online RL)", dqn_agent)],
        trained_agents_by_seed=trained_agents_by_seed,
    )
    summary_df = aggregate_experiment_results(raw_df)
    paired_df = compute_paired_cycle_comparisons(raw_df, reference_strategy="Q-Learning")

    raw_df.to_csv("results/raw_results.csv", index=False)
    summary_df.to_csv("results/summary.csv", index=False)
    paired_df.to_csv("results/paired_synthetic_comparison.csv", index=False)
    return raw_df, summary_df


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 3: Budget Sensitivity (Leak-Free Training History)
# ──────────────────────────────────────────────────────────────────────────────

def run_budget_sensitivity_experiment(
    agent: LinearQLearningAgent,
    dqn_agent: DuelingDQNAgent,
    base_seed: int = 42,
    budget_ratios: List[float] = [0.20, 0.35, 0.50, 0.65, 0.80],
    num_builds: int = 50,
    test_history: Optional[Dict[Any, float]] = None,
) -> pd.DataFrame:
    """
    Evaluates how each strategy scales as testing budget increases/decreases.
    CRITICAL FIX: Historical failure rates are derived strictly from `test_history`
    (the training set) or a separately generated training split `train_builds_b`,
    NEVER from the held-out evaluation `builds`.
    """
    print(f"\n[3/8] Running Budget Sensitivity Experiment over ratios: {budget_ratios}...")
    records = []

    for b_ratio in budget_ratios:
        bg_eval = BuildGenerator(num_tests=30, num_features=12, min_bugs=5, max_bugs=8, budget_ratio=b_ratio, seed=base_seed + 500)
        builds = bg_eval.generate_dataset(num_builds=num_builds, prefix=f"budget_eval_{int(b_ratio*100)}", base_seed=base_seed + 6000)

        if test_history is not None:
            th = dict(test_history)
        else:
            bg_tr = BuildGenerator(num_tests=30, num_features=12, min_bugs=5, max_bugs=8, budget_ratio=b_ratio, seed=base_seed)
            train_builds_b = bg_tr.generate_dataset(num_builds=60, prefix=f"budget_train_{int(b_ratio*100)}", base_seed=base_seed)
            th = _build_synthetic_test_history(train_builds_b)

        agent.set_test_history(dict(th))
        dqn_agent.set_test_history(dict(th))
        g_rec = GreedyRecencyCostBaseline(test_history=dict(th))

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

            g_rec_res = evaluate_policy_on_build(g_rec, env, build, is_rl=False, online_adapt=True)
            g_rec_res["strategy"] = "Greedy (Recency/Cost)"
            g_rec_res["budget_ratio"] = b_ratio
            records.append(g_rec_res)

            ql_res = evaluate_policy_on_build(agent, env, build, is_rl=True)
            ql_res["strategy"] = "Q-Learning (Linear)"
            ql_res["budget_ratio"] = b_ratio
            records.append(ql_res)

            dqn_res = evaluate_policy_on_build(dqn_agent, env, build, is_rl=True)
            dqn_res["strategy"] = "Dueling DQN (Online RL)"
            dqn_res["budget_ratio"] = b_ratio
            records.append(dqn_res)

    df_budget = pd.DataFrame(records)
    df_budget.to_csv("results/budget_sensitivity.csv", index=False)
    return df_budget


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 4: Reward Ablation (Unclamped, Pure RL Learning)
# ──────────────────────────────────────────────────────────────────────────────

def run_reward_ablation_experiment(
    train_builds: List[SoftwareBuild],
    test_builds: List[SoftwareBuild],
    episodes: int = 350,
    seed: int = 42,
    test_history: Optional[Dict[Any, float]] = None,
) -> pd.DataFrame:
    """
    Controlled ablation of reward function weights:
    - R1: Balanced (w_b=3.0, w_c=0.8, w_e=0.2)
    - R2: Bug-Only (w_b=5.0, w_c=0.0, w_e=0.0)
    - R3: Coverage-Only (w_b=0.0, w_c=5.0, w_e=0.2)
    - R4: Cost-Penalized (w_b=0.5, w_c=0.2, w_e=5.0)
    """
    print("\n[4/8] Running Reward Function Ablation Study...")
    ablation_configs = [
        ("Balanced (w_b=3.0, w_c=0.8, w_e=0.2)", 3.0, 0.8, 0.2),
        ("Bug-Only (w_b=5.0, w_c=0.0, w_e=0.0)", 5.0, 0.0, 0.0),
        ("Coverage-Only (w_b=0.0, w_c=5.0, w_e=0.2)", 0.0, 5.0, 0.2),
        ("Cost-Penalized (w_b=0.5, w_c=0.2, w_e=5.0)", 0.5, 0.2, 5.0),
    ]

    records = []
    for name, wb, wc, we in ablation_configs:
        print(f"  Training reward ablation variant: {name}...")
        agent, _ = train_agent(
            train_builds=train_builds,
            num_episodes=episodes,
            w_b=wb,
            w_c=wc,
            w_e=we,
            shaping_coef=0.0,
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
# Experiment 5: State Representation Ablation
# ──────────────────────────────────────────────────────────────────────────────

def run_state_ablation_experiment(
    train_builds: List[SoftwareBuild],
    test_builds: List[SoftwareBuild],
    episodes: int = 350,
    seed: int = 42,
    test_history: Optional[Dict[Any, float]] = None,
) -> pd.DataFrame:
    """
    Evaluates the contribution of different state-action feature groups:
    1. Full 12-Feature State
    2. No Recency Features (masks recent_fail_rate and last_verdict_fail)
    3. No Coverage Features (masks new_cov_norm, cov_per_cost, uncovered_risk_per_cost, coverage_progress)
    4. Cost & Budget Only (masks all failure and coverage features)
    """
    print("\n[5/8] Running State Feature Ablation Study...")
    full_mask = np.ones(NUM_FEATURES, dtype=np.float64)

    no_recency_mask = np.ones(NUM_FEATURES, dtype=np.float64)
    no_recency_mask[[1, 3]] = 0.0

    no_coverage_mask = np.ones(NUM_FEATURES, dtype=np.float64)
    no_coverage_mask[[4, 5, 6, 10]] = 0.0

    cost_only_mask = np.zeros(NUM_FEATURES, dtype=np.float64)
    cost_only_mask[[7, 8, 9]] = 1.0

    state_variants = [
        ("Full State (12 Features)", full_mask),
        ("No Recency Features", no_recency_mask),
        ("No Coverage Features", no_coverage_mask),
        ("Cost & Budget Only", cost_only_mask),
    ]

    records = []
    for label, mask in state_variants:
        print(f"  Training state variant: {label}...")
        agent, _ = train_agent(
            train_builds=train_builds,
            num_episodes=episodes,
            test_history=test_history,
            seed=seed,
            use_linear=True,
            shaping_coef=0.0,
            feature_mask=mask,
            verbose_interval=9999,
        )
        env = TestingEnvironment(test_builds[0])
        for build in test_builds[:60]:
            res = evaluate_policy_on_build(agent, env, build, is_rl=True)
            res["state_variant"] = label
            records.append(res)

    df_state = pd.DataFrame(records)
    df_state.to_csv("results/state_ablation.csv", index=False)
    return df_state


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 6: Agent Architecture Ablation
# ──────────────────────────────────────────────────────────────────────────────

def run_agent_ablation_experiment(
    train_builds: List[SoftwareBuild],
    test_builds: List[SoftwareBuild],
    episodes: int = 350,
    seed: int = 42,
    test_history: Optional[Dict[Any, float]] = None,
) -> pd.DataFrame:
    """
    Compares Dueling DQN vs. Linear Q-Learning vs. Contextual Bandit (LinUCB)
    vs. Discrete-State Tabular Q-Learning.
    """
    print("\n[6/8] Running Agent Architecture Ablation...")
    records = []

    variants = [
        ("dueling_dqn", "Dueling DQN (Online RL)"),
        ("linear", "Linear Q-Learning (ours)"),
        ("bandit", "Contextual Bandit (LinUCB)"),
        ("tabular", "Tabular Q-Learning"),
    ]

    for agent_type, label in variants:
        print(f"  Training {label}...")
        agent, _ = train_agent(
            train_builds=train_builds,
            num_episodes=episodes,
            test_history=test_history,
            seed=seed,
            agent_type=agent_type,
            shaping_coef=0.0,
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
# Experiment 7: Real-World CI Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_real_world_ci_experiment(
    dataset_name: str = "paintcontrol",
    episodes: int = 300,
    seeds: List[int] = [42, 101, 777],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates Dueling DQN, Linear Q-learning, and baselines (including symmetric
    Greedy Recency/Cost) on real-world ABB CI datasets.
    Explicitly reports total held-out cycles vs. defect-containing cycles, and uses
    mild test-name token proxy diversity weight (w_c=0.35) alongside fault detection.
    """
    print(f"\n[7/8] Running Real-World Evaluation on {dataset_name.upper()} CI dataset...")
    df = load_ci_dataset(dataset_name)
    train_cycles, test_cycles = split_ci_cycles(df, train_ratio=0.70)

    all_test_builds = RealWorldCIAdapter.load_all_builds_from_df(
        df, test_cycles, budget_ratio=0.45, include_zero_failure_cycles=True
    )
    train_builds = RealWorldCIAdapter.load_all_builds_from_df(
        df, train_cycles, budget_ratio=0.45, include_zero_failure_cycles=False
    )
    test_builds = [b for b in all_test_builds if b.num_hidden_bugs > 0]
    excluded_zero_fail = len(all_test_builds) - len(test_builds)

    print(
        f"  {dataset_name.upper()} Cycle Split: {len(train_cycles)} total train cycles "
        f"({len(train_builds)} with failures); {len(test_cycles)} total held-out test cycles "
        f"({len(test_builds)} evaluated with failures, {excluded_zero_fail} zero-failure cycles excluded "
        f"because BDR/APFD denominators require m >= 1 fault)."
    )

    test_history = _build_ci_test_history(df, train_cycles)

    trained_ci_by_seed: Dict[int, Dict[str, Any]] = {}
    primary_lin = None
    primary_dqn = None

    for s in seeds:
        ci_lin, _ = train_agent(
            train_builds=train_builds,
            num_episodes=episodes,
            test_history=test_history,
            seed=s,
            use_linear=True,
            agent_type="linear",
            w_b=3.0,
            w_c=0.35,
            w_e=0.25,
            shaping_coef=0.0,
            alpha=0.015,
            gamma=0.95,
            epsilon_min=0.02,
            verbose_interval=300,
        )
        ci_dqn, _ = train_agent(
            train_builds=train_builds,
            num_episodes=episodes,
            test_history=test_history,
            seed=s,
            agent_type="dueling_dqn",
            w_b=3.0,
            w_c=0.35,
            w_e=0.25,
            shaping_coef=0.0,
            alpha=0.008,
            gamma=0.95,
            epsilon_min=0.02,
            verbose_interval=300,
        )
        trained_ci_by_seed[s] = {
            "Q-Learning": ci_lin,
            "Dueling DQN (Online RL)": ci_dqn,
        }
        if primary_lin is None:
            primary_lin = ci_lin
            primary_dqn = ci_dqn

    raw_ci_df = run_benchmark_comparison(
        primary_lin,
        test_builds,
        seeds=seeds,
        test_history=test_history,
        extra_agents=[("Dueling DQN (Online RL)", primary_dqn)],
        trained_agents_by_seed=trained_ci_by_seed,
    )
    summary_ci_df = aggregate_experiment_results(raw_ci_df)
    paired_ci_df = compute_paired_cycle_comparisons(raw_ci_df, reference_strategy="Q-Learning")

    raw_ci_df.to_csv(f"results/real_world_{dataset_name}_raw.csv", index=False)
    summary_ci_df.to_csv(f"results/real_world_{dataset_name}_summary.csv", index=False)
    paired_ci_df.to_csv(f"results/real_world_{dataset_name}_paired.csv", index=False)

    print(f"\n  {dataset_name.upper()} Summary (Failing-Test Recall & NAPFD):")
    for _, row in summary_ci_df.iterrows():
        print(
            f"    {row['strategy']:30s} BDR={row['bug_detection_rate_mean']:.4f} (±{row['bug_detection_rate_ci95']:.4f})  "
            f"NAPFD={row['napfd_mean']:.4f}  APFD={row['apfd_mean']:.4f}  Bugs/Cost={row['bugs_per_cost_unit_mean']:.4f}"
        )

    return raw_ci_df, summary_ci_df


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 8: Plot Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_plots():
    """Generates all publication-grade figures in results/figures/."""
    print("\n[8/8] Generating Publication Figures in results/figures/...")
    os.makedirs("results/figures", exist_ok=True)

    PALETTE = {
        "Random": "#7f7f7f",
        "Greedy (Coverage/Cost)": "#ff7f0e",
        "Greedy (Failure/Cost)": "#d62728",
        "Greedy (Recency/Cost)": "#e377c2",
        "Q-Learning": "#2ca02c",
        "Q-Learning (Linear)": "#2ca02c",
        "Linear Q-Learning (ours)": "#2ca02c",
        "Dueling DQN (Online RL)": "#1f77b4",
        "Contextual Bandit (LinUCB)": "#17becf",
        "Tabular Q-Learning": "#9467bd",
        "Tabular Q-Learning (legacy)": "#9467bd",
    }

    # 1. Training Curve
    if os.path.exists("results/training_history.csv"):
        th = pd.read_csv("results/training_history.csv")
        ep_grp = th.groupby("episode")[["reward", "bug_detection_rate"]].mean().reset_index()
        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        window = 30
        ep_grp["reward_smooth"] = ep_grp["reward"].rolling(window, min_periods=1).mean()
        ep_grp["bdr_smooth"] = ep_grp["bug_detection_rate"].rolling(window, min_periods=1).mean() * 100

        color = "#1f77b4"
        ax1.set_xlabel("Training Episode", fontweight="bold")
        ax1.set_ylabel("Episode Reward", color=color, fontweight="bold")
        ax1.plot(ep_grp["episode"], ep_grp["reward_smooth"], color=color, linewidth=2, label="Reward (Smoothed)")
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color2 = "#2ca02c"
        ax2.set_ylabel("Bug Detection Rate (%)", color=color2, fontweight="bold")
        ax2.plot(ep_grp["episode"], ep_grp["bdr_smooth"], color=color2, linewidth=2, linestyle="--", label="BDR (%)")
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(0, 105)

        plt.title("TestPilot-RL: Multi-Seed Training Dynamics and Convergence", pad=12, fontweight="bold")
        fig.tight_layout()
        plt.savefig("results/figures/training_curve.png", dpi=300)
        plt.close()
        print("  Saved results/figures/training_curve.png")

    # 2. Baseline Comparison (synthetic)
    if os.path.exists("results/summary.csv"):
        summ = pd.read_csv("results/summary.csv")
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))

        colors = [PALETTE.get(s, "#1f77b4") for s in summ["strategy"]]
        metrics = [
            ("bug_detection_rate_mean", "bug_detection_rate_ci95", "Bug Detection Rate (BDR)", "Mean BDR", (0, 1.08)),
            ("coverage_rate_mean", "coverage_rate_ci95", "Code Feature Coverage", "Mean Coverage Ratio", (0, 1.08)),
            ("napfd_mean", "napfd_ci95", "NAPFD (Normalized Early Fault Detection)", "NAPFD Score", (0, 1.08)),
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
            ax.tick_params(axis="x", rotation=24)
            for bar, strat in zip(bars, summ["strategy"]):
                if "Q-Learning" in str(strat) or "DQN" in str(strat):
                    bar.set_edgecolor("#000000")
                    bar.set_linewidth(1.8)

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
            ("napfd", "NAPFD Score"),
        ]):
            for strat, grp in bs.groupby("strategy"):
                mean = grp.groupby("budget_ratio")[metric].mean()
                std = grp.groupby("budget_ratio")[metric].std()
                is_rl_s = ("Q-Learning" in strat) or ("DQN" in strat)
                lw = 2.5 if is_rl_s else 1.8
                ls = "-" if is_rl_s else "--"
                ax.plot(mean.index, mean.values, marker="o", label=strat,
                        color=strat_colors.get(strat, "#888"),
                        linewidth=lw, linestyle=ls)
                ax.fill_between(mean.index, mean - 0.5 * std, mean + 0.5 * std, alpha=0.10,
                                color=strat_colors.get(strat, "#888"))

            ax.set_xlabel("Testing Budget Ratio (fraction of full suite cost)", fontweight="bold")
            ax.set_ylabel(ylabel, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8.5)

        axes[0].set_title("Budget Sensitivity: Bug Detection Rate", fontweight="bold")
        axes[1].set_title("Budget Sensitivity: NAPFD Score", fontweight="bold")
        fig.suptitle("TestPilot-RL: Performance vs. Testing Budget", fontweight="bold")
        fig.tight_layout()
        plt.savefig("results/figures/budget_sensitivity.png", dpi=300)
        plt.close()
        print("  Saved results/figures/budget_sensitivity.png")

    # 4. Reward Ablation
    if os.path.exists("results/reward_ablation.csv"):
        ra = pd.read_csv("results/reward_ablation.csv")
        fig, ax = plt.subplots(figsize=(9.5, 5))
        ra_mean = ra.groupby("ablation_variant", sort=False)[["bug_detection_rate", "coverage_rate", "napfd"]].mean().reset_index()
        ra_melted = pd.melt(ra_mean, id_vars=["ablation_variant"],
                            value_vars=["bug_detection_rate", "coverage_rate", "napfd"],
                            var_name="Metric", value_name="Score")
        metric_labels = {"bug_detection_rate": "BDR", "coverage_rate": "Coverage", "napfd": "NAPFD"}
        ra_melted["Metric"] = ra_melted["Metric"].map(metric_labels)

        sns.barplot(data=ra_melted, x="ablation_variant", y="Score", hue="Metric",
                    ax=ax, palette=["#1f77b4", "#2ca02c", "#ff7f0e"])
        ax.set_title("Reward Formulation Ablation Study", fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_xlabel("Reward Weight Variant")
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="x", rotation=18)
        ax.legend(title="Metric")
        fig.tight_layout()
        plt.savefig("results/figures/reward_ablation.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved results/figures/reward_ablation.png")

    # 5. Agent Ablation
    if os.path.exists("results/agent_ablation.csv"):
        aa = pd.read_csv("results/agent_ablation.csv")
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 5))
        aa_mean = aa.groupby("agent_variant", sort=False)[["bug_detection_rate", "coverage_rate", "napfd"]].mean().reset_index()

        for ax, col, title in zip(
            axes,
            ["bug_detection_rate", "coverage_rate", "napfd"],
            ["Bug Detection Rate (BDR)", "Coverage Rate", "NAPFD Score"],
        ):
            colors = [PALETTE.get(v, "#1f77b4") for v in aa_mean["agent_variant"]]
            ax.bar(aa_mean["agent_variant"], aa_mean[col], color=colors, edgecolor="black", linewidth=0.8)
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.1)
            ax.tick_params(axis="x", rotation=22)

        fig.suptitle("Agent Architecture Ablation: Dueling DQN vs. Linear Q vs. LinUCB Bandit vs. Tabular Q", fontweight="bold")
        fig.tight_layout()
        plt.savefig("results/figures/agent_ablation.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved results/figures/agent_ablation.png")

    # 6. Real-World Evaluation — both datasets (plotting BDR and NAPFD)
    for dataset, title_suffix in [
        ("paintcontrol", "ABB Paint Control"),
        ("iofrol", "ABB IOF/ROL"),
    ]:
        path = f"results/real_world_{dataset}_summary.csv"
        if os.path.exists(path):
            ci_summ = pd.read_csv(path)
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
            colors = [PALETTE.get(s, "#1f77b4") for s in ci_summ["strategy"]]

            for ax, (col_mean, col_ci, ylabel) in zip(axes, [
                ("bug_detection_rate_mean", "bug_detection_rate_ci95", "Failing-Test Recall (BDR)"),
                ("napfd_mean", "napfd_ci95", "Normalized APFD (NAPFD)"),
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
                ax.set_ylim(0, 1.08)
                ax.tick_params(axis="x", rotation=24)
                for bar, strat in zip(bars, ci_summ["strategy"]):
                    if "Q-Learning" in str(strat) or "DQN" in str(strat):
                        bar.set_edgecolor("#000000")
                        bar.set_linewidth(1.8)

            axes[0].set_title("Failing-Test Recall (BDR)", fontweight="bold")
            axes[1].set_title("Normalized APFD (NAPFD)", fontweight="bold")
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

    train_seeds = [42, 101, 777]

    # ── Synthetic dataset setup ──
    print("Setting up synthetic training and test builds...")
    bg_train = BuildGenerator(num_tests=30, num_features=12, min_bugs=5, max_bugs=8, budget_ratio=0.45, seed=42)
    bg_test = BuildGenerator(num_tests=30, num_features=12, min_bugs=5, max_bugs=8, budget_ratio=0.45, seed=99)
    train_builds = bg_train.generate_dataset(num_builds=80, prefix="train_build", base_seed=42)
    test_builds = bg_test.generate_dataset(num_builds=120, prefix="heldout_build", base_seed=5000)

    syn_hist = _build_synthetic_test_history(train_builds)

    # ── Run experiments ──
    agent, dqn_agent, trained_by_seed, history_df = run_training_experiment(
        train_builds, episodes=450, seeds=train_seeds, test_history=syn_hist
    )
    raw_df, summary_df = run_main_comparison_experiment(
        agent, dqn_agent, test_builds, seeds=train_seeds, test_history=syn_hist, trained_agents_by_seed=trained_by_seed
    )

    print("\n===== SYNTHETIC BENCHMARK SUMMARY =====")
    for _, row in summary_df.iterrows():
        print(
            f"  {row['strategy']:30s} BDR={row['bug_detection_rate_mean']:.4f} (±{row['bug_detection_rate_ci95']:.4f})  "
            f"NAPFD={row['napfd_mean']:.4f}  APFD={row['apfd_mean']:.4f}  Bugs/Cost={row['bugs_per_cost_unit_mean']:.4f}"
        )

    df_budget = run_budget_sensitivity_experiment(agent, dqn_agent, base_seed=42, num_builds=50, test_history=syn_hist)
    df_ablation = run_reward_ablation_experiment(train_builds, test_builds, episodes=300, seed=42, test_history=syn_hist)
    df_state = run_state_ablation_experiment(train_builds, test_builds, episodes=300, seed=42, test_history=syn_hist)
    df_agent_ablation = run_agent_ablation_experiment(train_builds, test_builds, episodes=300, seed=42, test_history=syn_hist)

    # Real-world experiments across multiple training seeds
    run_real_world_ci_experiment(dataset_name="paintcontrol", episodes=300, seeds=train_seeds)
    run_real_world_ci_experiment(dataset_name="iofrol", episodes=300, seeds=train_seeds)

    # Generate plots
    generate_all_plots()

    print("\n=======================================================")
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=======================================================")
    print("\nKey result files:")
    print("  results/summary.csv                         — synthetic benchmark comparison")
    print("  results/state_ablation.csv                  — state feature group ablation")
    print("  results/real_world_paintcontrol_summary.csv — PaintControl CI results")
    print("  results/real_world_iofrol_summary.csv       — IOF/ROL CI results")
    print("  results/figures/                            — all publication figures")


if __name__ == "__main__":
    main()