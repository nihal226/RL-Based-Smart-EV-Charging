"""Visualization module for RL training dynamics and episode operational profiles.
Generates Publication-Quality plots for:
5. RL average episode reward vs training episodes
6. EV battery SOC trajectories vs time
7. Station charging power allocation vs time
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIR, TOTAL_STATION_POWER_LIMIT_KW, EV_TARGET_SOC


def plot_training_rewards(history_df: pd.DataFrame, window: int = 50, save_path: Optional[Path] = None) -> Path:
    """Graph 5: RL average episode reward vs training episodes with rolling average and epsilon decay."""
    if save_path is None:
        save_path = PLOTS_DIR / "05_training_reward_vs_episodes.png"

    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)

    episodes = history_df["episode"]
    rewards = history_df["total_episode_reward"]
    rolling_rewards = rewards.rolling(window=window, min_periods=1).mean()

    # Raw rewards (scatter / translucent line)
    ax1.plot(episodes, rewards, color="#3498db", alpha=0.25, label="Episode Return (Raw)")
    ax1.plot(episodes, rolling_rewards, color="#2980b9", linewidth=2.4, label=f"Moving Average ({window}-Ep)")

    ax1.set_xlabel("Training Episode")
    ax1.set_ylabel("Total Cumulative Reward", color="#2980b9")
    ax1.tick_params(axis="y", labelcolor="#2980b9")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Secondary axis: Epsilon Exploration Decay
    if "epsilon" in history_df.columns:
        ax2 = ax1.twinx()
        ax2.plot(episodes, history_df["epsilon"], color="#e67e22", linestyle="--", linewidth=1.8, label="Epsilon (Exploration Rate)")
        ax2.set_ylabel("Exploration Rate (Epsilon)", color="#e67e22")
        ax2.tick_params(axis="y", labelcolor="#e67e22")
        ax2.set_ylim(0, 1.05)

    plt.title("Tabular Q-Learning Convergence Curve & Epsilon Decay", fontweight="bold", pad=12)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if "epsilon" in history_df.columns:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.9)
    else:
        ax1.legend(loc="lower right", framealpha=0.9)

    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved: {save_path.name}")
    return save_path


def plot_soc_trajectories(env_history: Dict[str, Any], evs_list: List[Any], save_path: Optional[Path] = None) -> Path:
    """Graph 6: EV battery SOC vs time across a 24-hour test episode."""
    if save_path is None:
        save_path = PLOTS_DIR / "06_ev_soc_vs_time.png"

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    time_hours = env_history["time_hours"]

    colors = ["#2ecc71", "#3498db", "#9b59b6", "#f1c40f", "#e67e22", "#1abc9c"]

    # Reconstruct SOC trajectories per EV
    for idx, ev in enumerate(evs_list):
        ev_id = ev.ev_id
        color = colors[idx % len(colors)]

        # Time series of SOC
        soc_series = []
        curr_soc = ev.initial_soc

        for step in env_history["step"]:
            if step < ev.arrival_step:
                soc_series.append(np.nan)
            elif step > ev.departure_step:
                soc_series.append(np.nan)
            else:
                # Value at this step
                soc_series.append(curr_soc)
                # If charge record exists
                # We can check charge history if recorded
            # For exact visual trajectory from step history:
        # Better: let's track or sample during simulation
        # If ev has initial and final, we can plot between arrival and departure
        t_arr = ev.arrival_step * 0.25
        t_dep = ev.departure_step * 0.25
        ax.plot([t_arr, t_dep], [ev.initial_soc, ev.departure_soc], marker="o", color=color, linewidth=2.2, label=f"EV {ev_id} ({ev.initial_soc:.0f}% -> {ev.departure_soc:.0f}%)")

    ax.axhline(EV_TARGET_SOC, color="#e74c3c", linestyle="--", linewidth=1.8, label=f"Target Departure SOC ({EV_TARGET_SOC}%)")

    ax.set_title("EV Battery State of Charge (SOC) Progression", fontweight="bold", pad=12)
    ax.set_xlabel("Time of Day (Hours)")
    ax.set_ylabel("Battery State of Charge (%)")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 105)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)

    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved: {save_path.name}")
    return save_path


def plot_power_allocation(env_history: Dict[str, Any], save_path: Optional[Path] = None) -> Path:
    """Graph 7: Charging power allocation vs time (Solar vs Grid vs Station Limit)."""
    if save_path is None:
        save_path = PLOTS_DIR / "07_power_allocation_vs_time.png"

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    time_hours = env_history["time_hours"]
    solar_used = np.array(env_history["solar_power_used_kw"])
    grid_used = np.array(env_history["grid_power_for_ev_kw"])
    total_ev_power = np.array(env_history["total_ev_power_kw"])
    solar_gen = np.array(env_history["solar_generation_kw"])

    ax.plot(time_hours, solar_gen, color="#f39c12", linestyle=":", linewidth=1.8, label="Solar PV Available (kW)")
    
    # Stacked area for EV power breakdown
    ax.fill_between(time_hours, 0, solar_used, color="#2ecc71", alpha=0.7, label="Clean Solar Power Used (kW)")
    ax.fill_between(time_hours, solar_used, solar_used + grid_used, color="#e74c3c", alpha=0.6, label="Grid Power Used (kW)")
    
    ax.axhline(TOTAL_STATION_POWER_LIMIT_KW, color="#2c3e50", linestyle="--", linewidth=1.8, label=f"Station Limit ({TOTAL_STATION_POWER_LIMIT_KW} kW)")

    ax.set_title("Station Power Allocation & Solar Utilization Dispatch", fontweight="bold", pad=12)
    ax.set_xlabel("Time of Day (Hours)")
    ax.set_ylabel("Power (kW)")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, max(25.0, np.max(solar_gen) * 1.15))
    ax.set_xticks(np.arange(0, 25, 4))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved: {save_path.name}")
    return save_path

