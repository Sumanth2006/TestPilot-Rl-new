"""
Training pipeline for TestPilot-RL Tabular Q-Learning Agent.

Trains on training scenarios with epsilon decay, Bellman updates,
and legal action masking. Saves model checkpoint and training history.
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from src.simulator import BuildGenerator, RealWorldCIAdapter, SoftwareBuild
from src.environment import TestingEnvironment
from src.q_learning import TabularQLearningAgent
from src.data_loader import load_ci_dataset, split_ci_cycles


def train_agent(
    train_builds: List[SoftwareBuild],
    num_episodes: int = 1500,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    w_b: float = 2.5,
    w_c: float = 0.8,
    w_e: float = 0.3,
    budget_bins: int = 5,
    coverage_bins: int = 5,
    history_bins: int = 3,
    seed: int = 42,
    verbose_interval: int = 250,
) -> Tuple[TabularQLearningAgent, pd.DataFrame]:
    """
    Executes the Tabular Q-learning training loop across training builds.
    """
    rng = np.random.default_rng(seed)
    max_tests = max(b.num_tests for b in train_builds)

    agent = TabularQLearningAgent(
        num_actions=max_tests,
        budget_bins=budget_bins,
        coverage_bins=coverage_bins,
        history_bins=history_bins,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )

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

    print(f"Starting Q-Learning training for {num_episodes} episodes across {len(train_builds)} builds...")

    for ep in range(1, num_episodes + 1):
        # Sample or cycle through training builds
        build = train_builds[ep % len(train_builds)]
        state, info = env.reset(build)

        episode_reward = 0.0
        step = 0
        td_errors = []

        while True:
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break

            action = agent.select_action(state, legal_actions, explore=True)
            next_state, reward, terminated, truncated, step_info = env.step(action)
            episode_reward += reward

            next_legal_actions = env.get_legal_actions()
            td_err = agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_legal_actions=next_legal_actions,
                terminated=terminated,
            )
            td_errors.append(abs(td_err))
            state = next_state
            step += 1

            if terminated:
                break

        current_eps = agent.decay_epsilon()
        final_info = env._get_info()

        history_records.append({
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
        })

        if ep % verbose_interval == 0 or ep == num_episodes:
            recent_avg_r = np.mean([h["reward"] for h in history_records[-verbose_interval:]])
            recent_avg_bdr = np.mean([h["bug_detection_rate"] for h in history_records[-verbose_interval:]])
            print(
                f"Episode {ep:4d}/{num_episodes} | Avg Reward: {recent_avg_r:6.2f} | "
                f"BDR: {recent_avg_bdr*100:5.1f}% | Epsilon: {current_eps:.3f}"
            )

    history_df = pd.DataFrame(history_records)
    return agent, history_df


def main():
    parser = argparse.ArgumentParser(description="Train TestPilot-RL Q-Learning Agent")
    parser.add_argument("--episodes", type=int, default=1200, help="Number of training episodes")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "paintcontrol", "iofrol"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_path", type=str, default="models/q_table.npy", help="Path to save Q-table")
    parser.add_argument("--history_path", type=str, default="results/training_history.csv", help="History CSV path")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if args.dataset == "synthetic":
        bg = BuildGenerator(num_tests=30, num_features=12, min_bugs=5, max_bugs=8, budget_ratio=0.45, seed=args.seed)
        train_builds = bg.generate_dataset(num_builds=50, prefix="train_build", base_seed=args.seed)
    else:
        df = load_ci_dataset(args.dataset)
        train_cycles, _ = split_ci_cycles(df, train_ratio=0.7)
        train_builds = RealWorldCIAdapter.load_all_builds_from_df(df, train_cycles[:50], budget_ratio=0.45)
        if not train_builds:
            raise RuntimeError(f"No valid train builds extracted from {args.dataset}")

    agent, history_df = train_agent(
        train_builds=train_builds,
        num_episodes=args.episodes,
        seed=args.seed,
    )

    agent.save(args.save_path)
    history_df.to_csv(args.history_path, index=False)
    print(f"Training complete. Model saved to {args.save_path}. History saved to {args.history_path}.")


if __name__ == "__main__":
    main()
