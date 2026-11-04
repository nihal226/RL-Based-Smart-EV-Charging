"""Environmental sensitivity analysis: Evaluating system response under reduced solar availability.
Compares standard solar generation vs. a 30% solar reduction (cloudy/winter scenario)
across Immediate Charging, Solar-First Charging, and RL Q-Learning.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import (
    TEST_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    SOLAR_REDUCTION_FACTOR,
)
from environment.ev_environment import EVChargingEnv
from agents.q_learning import QLearningAgent
from baselines.immediate_charging import ImmediateChargingPolicy
from baselines.solar_first import SolarFirstChargingPolicy
from experiments.evaluate import evaluate_policy_on_episode


def run_solar_sensitivity_analysis(
    reduction_factor: float = SOLAR_REDUCTION_FACTOR,
    num_episodes: int = 20,
) -> pd.DataFrame:
    """Evaluates all three strategies under 100% nominal solar vs reduced solar."""
    print(f"\n==================================================")
    print(f"Running Solar Sensitivity Analysis (Solar Reduction = {int((1-reduction_factor)*100)}%)")
    print(f"==================================================")

    model_path = MODELS_DIR / "trained_q_table.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_path} not found. Please train agent first.")

    rl_agent = QLearningAgent(num_states=1296, num_actions=3)
    rl_agent.load(model_path)

    policies = {
        "Immediate Charging": ImmediateChargingPolicy(),
        "Solar-First Charging": SolarFirstChargingPolicy(),
        "RL (Q-Learning)": rl_agent,
    }

    test_profiles = sorted(list(TEST_DATA_DIR.glob("episode_*_profile.csv")))[:num_episodes]
    test_evs = sorted(list(TEST_DATA_DIR.glob("episode_*_evs.csv")))[:num_episodes]

    env = EVChargingEnv()
    records = []

    for scenario_name, factor in [("Nominal Solar (100%)", 1.0), ("Reduced Solar (70%)", reduction_factor)]:
        for strat_name, strat_obj in policies.items():
            cost_list = []
            grid_kwh_list = []
            solar_kwh_list = []
            solar_pct_list = []
            soc_pct_list = []
            peak_kw_list = []

            for prof_p, ev_p in zip(test_profiles, test_evs):
                prof_df = pd.read_csv(prof_p)
                prof_df["solar_generation_kw"] = prof_df["solar_generation_kw"] * factor

                # Load into environment
                env.profile_df = prof_df
                evs_df = pd.read_csv(ev_p)
                env.load_episode_from_csv(prof_p, ev_p)
                env.profile_df["solar_generation_kw"] = prof_df["solar_generation_kw"]

                m, _ = evaluate_policy_on_episode(env, strat_name, strat_obj)

                cost_list.append(m["total_grid_electricity_cost"])
                grid_kwh_list.append(m["total_grid_energy_kwh"])
                solar_kwh_list.append(m["total_solar_energy_kwh"])
                solar_pct_list.append(m["solar_contribution_pct"])
                soc_pct_list.append(m["target_soc_achievement_pct"])
                peak_kw_list.append(m["peak_grid_demand_kw"])

            records.append({
                "solar_condition": scenario_name,
                "strategy": strat_name,
                "avg_grid_cost_usd": round(float(np.mean(cost_list)), 2),
                "avg_grid_energy_kwh": round(float(np.mean(grid_kwh_list)), 2),
                "avg_solar_energy_kwh": round(float(np.mean(solar_kwh_list)), 2),
                "solar_contribution_pct": round(float(np.mean(solar_pct_list)), 2),
                "avg_peak_demand_kw": round(float(np.mean(peak_kw_list)), 2),
                "target_soc_satisfaction_pct": round(float(np.mean(soc_pct_list)), 2),
            })

    sensitivity_df = pd.DataFrame(records)
    out_path = METRICS_DIR / "solar_sensitivity_summary.csv"
    sensitivity_df.to_csv(out_path, index=False)
    print(f"\n[Solar Sensitivity] Saved results to {out_path}")
    print("\nSOLAR SENSITIVITY SUMMARY:")
    print(sensitivity_df.to_string(index=False))

    return sensitivity_df


if __name__ == "__main__":
    run_solar_sensitivity_analysis()

