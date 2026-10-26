"""
Training pipeline for TestPilot-RL Agents.

Supports:
- DuelingDQNAgent (Deep Dueling Double Q-Network with Online Adaptation)
- LinearQLearningAgent (12-feature Linear Approximate Q-Learning learned from zero/random init)
- ContextualBanditAgent (LinUCB Contextual Bandit)
- TabularQLearningAgent (Discrete-state Tabular Q-Learning)
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any

from src.simulator import BuildGenerator, RealWorldCIAdapter, SoftwareBuild
from src.environment import TestingEnvironment
from src.q_learning import (
    DuelingDQNAgent,
    LinearQLearningAgent,
    ContextualBanditAgent,
    TabularQLearningAgent,
    extract_test_features,
)
from src.data_loader import load_ci_dataset, split_ci_cycles


def build_canonical_ci_history(df: pd.DataFrame, train_cycles: List[int]) -> Dict[Any, float]:
    """
    Computes per-test historical failure rates strictly from `train_cycles` (no leakage).
    Stores BOTH canonical string `Name` keys and global integer `name_to_idx` keys
    so that all baselines and RL agents resolve the exact test identity without mismatch.
    """
    train_df = df[df["Cycle"].isin(train_cycles)]
    failure_rates = train_df.groupby("Name")["Verdict"].mean().to_dict()

    all_names = df["Name"].astype(str).unique().tolist()
    name_to_idx = {name: idx for idx, name in enumerate(all_names)}

    history: Dict[Any, float] = {}
    for name, rate in failure_rates.items():
        s_name = str(name)
        history[s_name] = float(rate)
        if s_name in name_to_idx:
            history[name_to_idx[s_name]] = float(rate)
    return history


def train_agent(
    train_builds: List[SoftwareBuild],
    num_episodes: int = 600,
    alpha: float = 0.012,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.02,
    epsilon_decay: float = 0.994,
    w_b: float = 3.0,
    w_c: float = 0.8,
    w_e: float = 0.2,
    budget_bins: int = 5,
    coverage_bins: int = 5,
    history_bins: int = 3,
    test_history: Optional[Dict[Any, float]] = None,
    w_fail: float = 0.0,
    w_cov: float = 0.0,
    seed: int = 42,
    verbose_interval: int = 200,
    use_linear: bool = True,
    agent_type: Optional[str] = None,
    shaping_coef: float = 0.0,
    replay_capacity: int = 8000,
    batch_size: int = 32,
    feature_mask: Optional[np.ndarray] = None,
) -> Tuple[Any, pd.DataFrame]:
    """Train a Reinforcement Learning agent across the supplied training builds."""

    if not train_builds:
        raise ValueError("train_builds must contain at least one build.")
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")
    if verbose_interval <= 0:
        raise ValueError("verbose_interval must be positive.")

    max_tests = max(build.num_tests for build in train_builds)
    history = dict(test_history) if test_history else {}

    if agent_type is None:
        resolved_type = "linear" if use_linear else "tabular"
    else:
        resolved_type = agent_type.lower()

    if resolved_type == "dueling_dqn":
        agent = DuelingDQNAgent(
            alpha=min(alpha, 0.006),
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            shaping_coef=shaping_coef,
            seed=seed,
            test_history=history,
            num_actions=max_tests,
        )
        needs_features = True
    elif resolved_type == "bandit":
        agent = ContextualBanditAgent(
            ucb_alpha=0.25,
            l2_reg=1.0,
            seed=seed,
            test_history=history,
        )
        needs_features = True
    elif resolved_type == "linear":
        agent = LinearQLearningAgent(
            alpha=alpha,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            test_history=history,
            shaping_coef=shaping_coef,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            seed=seed,
            num_actions=max_tests,
        )
        needs_features = True
    else:
        agent = TabularQLearningAgent(
            num_actions=max_tests,
            budget_bins=budget_bins,
            coverage_bins=coverage_bins,
            history_bins=history_bins,
            alpha=0.1,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            test_history=history,
            w_fail=w_fail,
            w_cov=w_cov,
            seed=seed,
        )
        needs_features = False

    if feature_mask is not None:
        agent.feature_mask = np.asarray(feature_mask, dtype=np.float64)

    env = TestingEnvironment(
        build=train_builds[0],
        w_b=w_b,
        w_c=w_c,
        w_e=w_e,
        budget_bins=budget_bins,
        coverage_bins=coverage_bins,
        history_bins=history_bins,
    )

    history_records = []

    print(
        f"Starting {resolved_type.upper()} training (seed={seed}) for {num_episodes} episodes "
        f"across {len(train_builds)} builds..."
    )

    for ep in range(1, num_episodes + 1):
        build = train_builds[(ep - 1) % len(train_builds)]
        state, info = env.reset(build)

        episode_reward = 0.0
        step = 0
        td_errors = []
        executed_order = []

        while True:
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break

            action = agent.select_action(
                state,
                legal_actions,
                explore=True,
                env=env,
            )

            action_features = None
            if needs_features:
                action_features = extract_test_features(
                    action,
                    env,
                    agent.test_history,
                )
                if feature_mask is not None:
                    action_features = action_features * agent.feature_mask

            next_state, reward, terminated, truncated, step_info = env.step(action)
            episode_reward += reward
            executed_order.append(action)

            next_legal_actions = env.get_legal_actions()

            if needs_features:
                td_err = agent.update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    next_legal_actions=next_legal_actions,
                    terminated=terminated or truncated,
                    env=env,
                    action_features=action_features,
                )
                if feature_mask is not None and hasattr(agent, "w"):
                    agent.w = agent.w * agent.feature_mask
            else:
                td_err = agent.update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    next_legal_actions=next_legal_actions,
                    terminated=terminated or truncated,
                    env=env,
                )

            td_errors.append(abs(float(td_err)))
            state = next_state
            step += 1

            if terminated or truncated:
                break

        if hasattr(agent, "adapt_online"):
            agent.adapt_online(build, executed_order)

        current_eps = agent.decay_epsilon()
        final_info = env._get_info()

        history_records.append(
            {
                "episode": ep,
                "reward": round(episode_reward, 3),
                "bug_detection_rate": round(final_info["bug_detection_rate"], 3),
                "unique_bugs_detected": final_info["unique_bugs_detected"],
                "coverage_rate": round(final_info["coverage_rate"], 3),
                "budget_consumed": final_info["budget_consumed"],
                "bugs_per_cost_unit": round(final_info["bugs_per_cost_unit"], 4),
                "steps": step,
                "epsilon": round(current_eps, 4),
                "mean_td_error": round(float(np.mean(td_errors)) if td_errors else 0.0, 4),
            }
        )

        if ep % verbose_interval == 0 or ep == num_episodes:
            recent_records = history_records[-verbose_interval:]
            recent_avg_reward = np.mean([r["reward"] for r in recent_records])
            recent_avg_bdr = np.mean([r["bug_detection_rate"] for r in recent_records])
            print(
                f"Episode {ep:4d}/{num_episodes} | "
                f"Avg Reward: {recent_avg_reward:6.2f} | "
                f"BDR: {recent_avg_bdr * 100:5.1f}% | "
                f"Epsilon: {current_eps:.3f}"
            )

    # Restore base training history so evaluation starts from clean pre-test state
    if test_history is not None:
        agent.set_test_history(dict(test_history))

    history_df = pd.DataFrame(history_records)
    return agent, history_df


def main():
    parser = argparse.ArgumentParser(description="Train TestPilot-RL Agent")
    parser.add_argument("--episodes", type=int, default=600, help="Number of training episodes")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "paintcontrol", "iofrol"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_path", type=str, default="models/scratch_model.npy", help="Path to save the model")
    parser.add_argument("--history_path", type=str, default="results/scratch/training_history.csv")
    parser.add_argument("--tabular", action="store_true", help="Use the tabular agent")
    parser.add_argument("--agent_type", type=str, default=None, choices=["dueling_dqn", "linear", "bandit", "tabular"])
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if args.dataset == "synthetic":
        generator = BuildGenerator(
            num_tests=30,
            num_features=12,
            min_bugs=5,
            max_bugs=8,
            budget_ratio=0.45,
            seed=args.seed,
        )
        train_builds = generator.generate_dataset(
            num_builds=80,
            prefix="train_build",
            base_seed=args.seed,
        )

        fail_counts = {test_id: 0 for test_id in range(30)}
        exec_counts = {test_id: 0 for test_id in range(30)}
        for build in train_builds:
            for test_id in build.test_cases:
                if build.bug_detection_map.get(test_id, set()):
                    fail_counts[test_id] += 1
                exec_counts[test_id] += 1

        test_history: Dict[Any, float] = {}
        for test_id in range(30):
            rate = fail_counts[test_id] / max(1, exec_counts[test_id])
            test_history[test_id] = rate
            test_history[f"SynTest_{test_id:02d}"] = rate
        w_c_val = 0.8
    else:
        df = load_ci_dataset(args.dataset)
        train_cycles, _ = split_ci_cycles(df, train_ratio=0.7)
        train_builds = RealWorldCIAdapter.load_all_builds_from_df(
            df,
            train_cycles,
            budget_ratio=0.45,
        )
        if not train_builds:
            raise RuntimeError(f"No valid training builds extracted from {args.dataset}")

        test_history = build_canonical_ci_history(df, train_cycles)
        w_c_val = 0.0  # Real-world CI logs have no ground-truth code coverage

    agent, history_df = train_agent(
        train_builds=train_builds,
        num_episodes=args.episodes,
        w_c=w_c_val,
        test_history=test_history,
        seed=args.seed,
        use_linear=not args.tabular,
        agent_type=args.agent_type,
    )

    if hasattr(agent, "save"):
        agent.save(args.save_path)

    history_dir = os.path.dirname(args.history_path)
    os.makedirs(history_dir or ".", exist_ok=True)
    history_df.to_csv(args.history_path, index=False)

    print(f"Training complete. Model saved to {args.save_path}.")
    print(f"Training history saved to {args.history_path}.")


if __name__ == "__main__":
    main()