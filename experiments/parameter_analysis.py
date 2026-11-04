"""Learning rate sensitivity analysis for Tabular Q-Learning.
Compares convergence dynamics, stability, and final performance across:
  alpha = 0.01 (slow, cautious updates)
  alpha = 0.10 (nominal baseline)
  alpha = 0.50 (aggressive updates)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from config import (
    METRICS_DIR,
    MODELS_DIR,
    LEARNING_RATE_EXPERIMENTS,
    DISCOUNT_FACTOR,
    INITIAL_EPSILON,
    MIN_EPSILON,
    EPSILON_DECAY,
    RANDOM_SEED,
)
from experiments.train import run_training
from experiments.evaluate import run_full_evaluation, evaluate_policy_on_episode
from environment.ev_environment import EVChargingEnv
from agents.q_learning import QLearningAgent


def run_learning_rate_experiments(
    learning_rates: List[float] = LEARNING_RATE_EXPERIMENTS,
    episodes_per_exp: int = 1200,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Trains separate agents with different learning rates and evaluates test performance."""
    print(f"\n==================================================")
    print(f"Running Parameter Experiments: Learning Rates {learning_rates}")
    print(f"Episodes per experiment: {episodes_per_exp}")
    print(f"==================================================")

    all_histories = []
    summary_results = []

    for lr in learning_rates:
        tag = f"alpha_{str(lr).replace('.', '_')}"
        model_name = f"model_{tag}.pkl"
        hist_name = f"history_{tag}.csv"

        print(f"\n>>> Running Experiment: alpha = {lr} <<<")
        agent, history_df = run_training(
            num_episodes=episodes_per_exp,
            learning_rate=lr,
            discount_factor=DISCOUNT_FACTOR,
            initial_epsilon=INITIAL_EPSILON,
            min_epsilon=MIN_EPSILON,
            epsilon_decay=EPSILON_DECAY,
            seed=seed,
            save_model=True,
            model_filename=model_name,
            history_filename=hist_name,
        )

        history_df["learning_rate"] = lr
        all_histories.append(history_df)

        # Evaluate on the 20 test episodes
        test_profiles = sorted(list(Path(PROJECT_ROOT / "data" / "testing").glob("episode_*_profile.csv")))[:20]
        test_evs = sorted(list(Path(PROJECT_ROOT / "data" / "testing").glob("episode_*_evs.csv")))[:20]

        env = EVChargingEnv()
        test_metrics_list = []
        for prof_p, ev_p in zip(test_profiles, test_evs):
            env.load_episode_from_csv(prof_p, ev_p)
            m, _ = evaluate_policy_on_episode(env, "RL (Q-Learning)", agent)
            test_metrics_list.append(m)

        summary_results.append({
            "learning_rate": lr,
            "final_100_train_reward": round(float(history_df["total_episode_reward"].tail(100).mean()), 2),
            "test_avg_cost_usd": round(float(np.mean([m["total_grid_electricity_cost"] for m in test_metrics_list])), 2),
            "test_avg_solar_kwh": round(float(np.mean([m["total_solar_energy_kwh"] for m in test_metrics_list])), 2),
            "test_solar_pct": round(float(np.mean([m["solar_contribution_pct"] for m in test_metrics_list])), 2),
            "test_avg_soc_satisfaction_pct": round(float(np.mean([m["target_soc_achievement_pct"] for m in test_metrics_list])), 2),
            "test_avg_unmet_kwh": round(float(np.mean([m["total_unmet_energy_kwh"] for m in test_metrics_list])), 2),
            "test_mean_reward": round(float(np.mean([m["total_episode_reward"] for m in test_metrics_list])), 2),
        })

    combined_history = pd.concat(all_histories, ignore_index=True)
    summary_df = pd.DataFrame(summary_results)

    combined_history.to_csv(METRICS_DIR / "lr_experiments_history.csv", index=False)
    summary_df.to_csv(METRICS_DIR / "lr_experiments_summary.csv", index=False)

    print(f"\n[LR Experiments] Saved summary results to {METRICS_DIR / 'lr_experiments_summary.csv'}")
    print("\nLEARNING RATE COMPARISON SUMMARY:")
    print(summary_df.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    run_learning_rate_experiments()

