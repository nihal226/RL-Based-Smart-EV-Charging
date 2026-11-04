"""Training pipeline for Tabular Q-Learning on Smart EV Charging.
Cycles across 80 training scenarios over 2000 episodes with epsilon decay,
tracking cumulative rewards, electricity costs, solar utilization, and unmet deficits.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from config import (
    TRAIN_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    TRAINING_EPISODES,
    LEARNING_RATE,
    DISCOUNT_FACTOR,
    INITIAL_EPSILON,
    MIN_EPSILON,
    EPSILON_DECAY,
    TIME_INTERVAL_HOURS,
    RANDOM_SEED,
)
from environment.ev_environment import EVChargingEnv
from agents.q_learning import QLearningAgent


def run_training(
    num_episodes: int = TRAINING_EPISODES,
    learning_rate: float = LEARNING_RATE,
    discount_factor: float = DISCOUNT_FACTOR,
    initial_epsilon: float = INITIAL_EPSILON,
    min_epsilon: float = MIN_EPSILON,
    epsilon_decay: float = EPSILON_DECAY,
    seed: int = RANDOM_SEED,
    save_model: bool = True,
    model_filename: str = "trained_q_table.pkl",
    history_filename: str = "training_history.csv",
) -> Tuple[QLearningAgent, pd.DataFrame]:
    """Executes the Q-Learning training loop across the training scenarios."""
    print(f"\n==================================================")
    print(f"Starting Q-Learning Training ({num_episodes} Episodes)")
    print(f"Alpha: {learning_rate}, Gamma: {discount_factor}, Initial Epsilon: {initial_epsilon}, Decay: {epsilon_decay}")
    print(f"==================================================")

    # Locate training files
    profile_files = sorted(list(TRAIN_DATA_DIR.glob("episode_*_profile.csv")))
    evs_files = sorted(list(TRAIN_DATA_DIR.glob("episode_*_evs.csv")))
    num_train_profiles = len(profile_files)

    if num_train_profiles == 0:
        raise FileNotFoundError(f"No training data found in {TRAIN_DATA_DIR}. Run data generation first.")

    # Initialize agent and environment
    agent = QLearningAgent(
        num_states=1296,
        num_actions=3,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        initial_epsilon=initial_epsilon,
        min_epsilon=min_epsilon,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )

    env = EVChargingEnv()
    rng = np.random.default_rng(seed)

    history_records: List[Dict[str, Any]] = []

    for ep in range(1, num_episodes + 1):
        # Pick scenario (cycle or random draw from training set)
        scenario_idx = (ep - 1) % num_train_profiles
        prof_path = profile_files[scenario_idx]
        ev_path = evs_files[scenario_idx]

        env.load_episode_from_csv(prof_path, ev_path)
        env.reset()

        episode_td_errors = []

        done = False
        while not done:
            eligible_evs = env.get_eligible_evs()
            actions_dict: Dict[int, int] = {}
            ev_states: Dict[int, int] = {}

            # Determine actions for eligible EVs using current Q-table & epsilon
            for ev in eligible_evs:
                state_tuple = env.discretize_state(ev)
                state_idx = env.state_to_index(state_tuple)
                action_idx = agent.select_action(state_idx, is_eval=False)

                actions_dict[ev.ev_id] = action_idx
                ev_states[ev.ev_id] = state_idx

            # Step environment forward
            step_info, total_reward, done, info = env.step(actions_dict)

            # Perform Tabular Q-learning update for each active EV
            # Compute credit-assigned reward components
            charge_recs = step_info["ev_charge_records"]
            tot_power = step_info["total_ev_power_kw"]
            solar_power_used = step_info["solar_power_used_kw"]
            grid_power_used = step_info["grid_power_for_ev_kw"]
            price = env.history["electricity_price"][-1]

            for ev in eligible_evs:
                ev_id = ev.ev_id
                s_idx = ev_states[ev_id]
                a_idx = actions_dict[ev_id]

                # Power share of this EV
                ev_power = charge_recs.get(ev_id, {}).get("power_drawn_kw", 0.0)
                power_ratio = (ev_power / tot_power) if tot_power > 0 else (1.0 / len(eligible_evs))

                ev_solar_kwh = solar_power_used * power_ratio * TIME_INTERVAL_HOURS
                ev_grid_kwh = grid_power_used * power_ratio * TIME_INTERVAL_HOURS
                ev_cost = ev_grid_kwh * price

                # Check if this EV departed in this step
                unmet_kwh = 0.0
                for comp_ev in info.get("completed_evs", []):
                    if comp_ev["ev_id"] == ev_id:
                        unmet_kwh = comp_ev["unmet_energy_kwh"]

                # Credit-assigned EV reward
                ev_reward_dict = env.reward_fn.calculate_step_reward(
                    grid_electricity_cost=ev_cost,
                    solar_energy_used_kwh=ev_solar_kwh,
                    total_grid_power_kw=env.history["total_grid_demand_kw"][-1],
                    unmet_energy_kwh=unmet_kwh,
                )
                ev_reward = ev_reward_dict["total_reward"]

                # Next state
                if done or ev.is_completed or not ev.is_eligible_to_charge(env.current_step):
                    next_s_idx = s_idx
                    ev_done = True
                else:
                    next_s_tuple = env.discretize_state(ev)
                    next_s_idx = env.state_to_index(next_s_tuple)
                    ev_done = False

                # Standard Q-learning Bellman update
                td_err = agent.update(s_idx, a_idx, ev_reward, next_s_idx, ev_done)
                episode_td_errors.append(abs(td_err))

        # Episode completed - decay exploration epsilon
        current_eps = agent.decay_epsilon()
        ep_metrics = env.get_episode_metrics()

        record = {
            "episode": ep,
            "epsilon": round(current_eps, 5),
            "total_episode_reward": ep_metrics["total_episode_reward"],
            "grid_cost": ep_metrics["total_grid_electricity_cost"],
            "grid_energy_kwh": ep_metrics["total_grid_energy_kwh"],
            "solar_energy_kwh": ep_metrics["total_solar_energy_kwh"],
            "solar_contribution_pct": ep_metrics["solar_contribution_pct"],
            "peak_grid_demand_kw": ep_metrics["peak_grid_demand_kw"],
            "avg_departure_soc": ep_metrics["avg_departure_soc"],
            "target_soc_achievement_pct": ep_metrics["target_soc_achievement_pct"],
            "total_unmet_energy_kwh": ep_metrics["total_unmet_energy_kwh"],
            "mean_td_error": round(float(np.mean(episode_td_errors)), 5) if episode_td_errors else 0.0,
        }
        history_records.append(record)

        # Console logging
        if ep == 1 or ep % 200 == 0 or ep == num_episodes:
            # 50-episode rolling average reward
            recent_rewards = [r["total_episode_reward"] for r in history_records[-min(ep, 50):]]
            avg_rew = float(np.mean(recent_rewards))
            recent_soc_pct = float(np.mean([r["target_soc_achievement_pct"] for r in history_records[-min(ep, 50):]]))
            print(
                f"Ep {ep:4d}/{num_episodes:4d} | "
                f"Eps: {current_eps:.4f} | "
                f"Reward (last 50): {avg_rew:7.2f} | "
                f"Cost: ${ep_metrics['total_grid_electricity_cost']:5.2f} | "
                f"Solar: {ep_metrics['total_solar_energy_kwh']:5.2f} kWh ({ep_metrics['solar_contribution_pct']:4.1f}%) | "
                f"SOC Satisfied: {recent_soc_pct:5.1f}%"
            )

    history_df = pd.DataFrame(history_records)

    # Persistence
    if save_model:
        model_path = MODELS_DIR / model_filename
        agent.save(model_path)
        print(f"\n[Training] Saved trained Q-table to {model_path}")

        csv_path = METRICS_DIR / history_filename
        history_df.to_csv(csv_path, index=False)
        print(f"[Training] Saved training metrics history to {csv_path}")

    return agent, history_df


if __name__ == "__main__":
    run_training()
