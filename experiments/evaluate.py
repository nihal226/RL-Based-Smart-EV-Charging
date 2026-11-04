"""Evaluation module for comparing RL Q-Learning against conventional baselines.
Evaluates all strategies under strictly identical test scenarios with frozen weights (zero updates).
Reports Mean, Std, Min, and Max for all 9 core academic performance metrics.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd

from config import (
    TEST_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    EVAL_EPISODES,
)
from environment.ev_environment import EVChargingEnv
from agents.q_learning import QLearningAgent
from baselines.immediate_charging import ImmediateChargingPolicy
from baselines.solar_first import SolarFirstChargingPolicy


def evaluate_policy_on_episode(
    env: EVChargingEnv,
    strategy_name: str,
    policy_obj: Any,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Runs a single 24-hour simulation episode using a given policy."""
    env.reset()
    done = False

    while not done:
        eligible_evs = env.get_eligible_evs()
        actions_dict: Dict[int, int] = {}

        if strategy_name == "RL (Q-Learning)":
            agent: QLearningAgent = policy_obj
            for ev in eligible_evs:
                s_tuple = env.discretize_state(ev)
                s_idx = env.state_to_index(s_tuple)
                # Frozen greedy selection with epsilon = 0
                action = agent.select_action(s_idx, is_eval=True)
                actions_dict[ev.ev_id] = action
        elif strategy_name == "Immediate Charging":
            actions_dict = policy_obj.get_actions(env)
        elif strategy_name == "Solar-First Charging":
            actions_dict = policy_obj.get_actions(env)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Advance step
        _, _, done, _ = env.step(actions_dict)

    metrics = env.get_episode_metrics()
    return metrics, env.history


def run_full_evaluation(
    model_path: Optional[Path] = None,
    num_test_episodes: int = EVAL_EPISODES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluates Immediate Charging, Solar-First Charging, and RL Q-Learning on identical test episodes."""
    if model_path is None:
        model_path = MODELS_DIR / "trained_q_table.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Train the model first.")

    # Load trained Q-Learning agent
    rl_agent = QLearningAgent(num_states=1296, num_actions=3)
    rl_agent.load(model_path)
    print(f"[Evaluation] Successfully loaded trained Q-Learning model from: {model_path}")

    # Initialize baselines
    immediate_policy = ImmediateChargingPolicy()
    solar_first_policy = SolarFirstChargingPolicy()

    strategies = {
        "Immediate Charging": immediate_policy,
        "Solar-First Charging": solar_first_policy,
        "RL (Q-Learning)": rl_agent,
    }

    test_profiles = sorted(list(TEST_DATA_DIR.glob("episode_*_profile.csv")))[:num_test_episodes]
    test_evs = sorted(list(TEST_DATA_DIR.glob("episode_*_evs.csv")))[:num_test_episodes]

    if len(test_profiles) < num_test_episodes:
        raise ValueError(f"Expected at least {num_test_episodes} test episodes in {TEST_DATA_DIR}, found {len(test_profiles)}")

    print(f"\n==================================================")
    print(f"Evaluating 3 Strategies on {len(test_profiles)} Independent Test Episodes")
    print(f"==================================================")

    detailed_results = []
    env = EVChargingEnv()

    for ep_idx, (prof_path, ev_path) in enumerate(zip(test_profiles, test_evs)):
        for strat_name, strat_obj in strategies.items():
            # Load exact identical episode
            env.load_episode_from_csv(prof_path, ev_path)
            metrics, history = evaluate_policy_on_episode(env, strat_name, strat_obj)

            row = {
                "episode": ep_idx + 1,
                "strategy": strat_name,
                "grid_cost_usd": metrics["total_grid_electricity_cost"],
                "grid_energy_kwh": metrics["total_grid_energy_kwh"],
                "solar_energy_kwh": metrics["total_solar_energy_kwh"],
                "solar_contribution_pct": metrics["solar_contribution_pct"],
                "peak_grid_demand_kw": metrics["peak_grid_demand_kw"],
                "avg_departure_soc_pct": metrics["avg_departure_soc"],
                "target_soc_achieved_pct": metrics["target_soc_achievement_pct"],
                "unmet_energy_kwh": metrics["total_unmet_energy_kwh"],
                "total_ev_energy_kwh": metrics["total_ev_energy_kwh"],
                "episode_reward": metrics["total_episode_reward"],
            }
            detailed_results.append(row)

    detailed_df = pd.DataFrame(detailed_results)

    # Calculate aggregated summary statistics (Mean, Std, Min, Max)
    metric_cols = [
        "grid_cost_usd",
        "grid_energy_kwh",
        "solar_energy_kwh",
        "solar_contribution_pct",
        "peak_grid_demand_kw",
        "avg_departure_soc_pct",
        "target_soc_achieved_pct",
        "unmet_energy_kwh",
        "total_ev_energy_kwh",
    ]

    summary_rows = []
    for strat_name in strategies.keys():
        strat_df = detailed_df[detailed_df["strategy"] == strat_name]
        for metric in metric_cols:
            summary_rows.append({
                "Strategy": strat_name,
                "Metric": metric,
                "Mean": round(float(strat_df[metric].mean()), 2),
                "Std": round(float(strat_df[metric].std()), 2),
                "Min": round(float(strat_df[metric].min()), 2),
                "Max": round(float(strat_df[metric].max()), 2),
            })

    summary_df = pd.DataFrame(summary_rows)

    # Save CSVs
    detailed_csv = METRICS_DIR / "test_episodes_detailed.csv"
    summary_csv = METRICS_DIR / "baseline_comparison.csv"
    detailed_df.to_csv(detailed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n[Evaluation] Saved detailed test results to {detailed_csv}")
    print(f"[Evaluation] Saved summary comparison to {summary_csv}")

    # Display clean comparison table
    pivot_mean = detailed_df.groupby("strategy")[metric_cols].mean().round(2).T
    print("\n" + "="*70)
    print("MEAN PERFORMANCE COMPARISON (20 Independent Test Episodes):")
    print("="*70)
    print(pivot_mean.to_string())
    print("="*70 + "\n")

    return detailed_df, summary_df


if __name__ == "__main__":
    run_full_evaluation()
