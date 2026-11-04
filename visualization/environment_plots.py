"""Visualization module for synthetic environment profiles.
Generates Publication-Quality plots for:
1. Synthetic solar generation vs time
2. Synthetic grid demand vs time
3. Electricity price vs time
4. EV arrival and departure schedule visualization
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

from config import PLOTS_DIR, TRAIN_DATA_DIR


# Matplotlib publication styling
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
})


def plot_solar_generation(profile_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Graph 1: Synthetic solar generation vs time."""
    if save_path is None:
        save_path = PLOTS_DIR / "01_solar_generation_vs_time.png"

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.plot(profile_df["time_hours"], profile_df["solar_generation_kw"], color="#f39c12", linewidth=2.2, label="Solar PV Generation (kW)")
    ax.fill_between(profile_df["time_hours"], 0, profile_df["solar_generation_kw"], color="#f39c12", alpha=0.25)

    ax.set_title("24-Hour Solar PV Power Generation Profile", fontweight="bold", pad=12)
    ax.set_xlabel("Time of Day (Hours)")
    ax.set_ylabel("Solar Power Generation (kW)")
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved: {save_path.name}")
    return save_path


def plot_grid_demand(profile_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Graph 2: Synthetic grid demand vs time."""
    if save_path is None:
        save_path = PLOTS_DIR / "02_grid_demand_vs_time.png"

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.plot(profile_df["time_hours"], profile_df["grid_demand_kw"], color="#2980b9", linewidth=2.2, label="Base Grid Demand (kW)")
    ax.axhline(35.0, color="#c0392b", linestyle=":", linewidth=1.8, label="Substation Peak Threshold (35 kW)")
    ax.fill_between(profile_df["time_hours"], 0, profile_df["grid_demand_kw"], color="#2980b9", alpha=0.15)

    ax.set_title("24-Hour Baseline Grid Demand Profile", fontweight="bold", pad=12)
    ax.set_xlabel("Time of Day (Hours)")
    ax.set_ylabel("Grid Demand (kW)")
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved: {save_path.name}")
    return save_path


def plot_electricity_price(profile_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Graph 3: Electricity price vs time."""
    if save_path is None:
        save_path = PLOTS_DIR / "03_electricity_price_vs_time.png"

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.step(profile_df["time_hours"], profile_df["electricity_price_per_kwh"], where="post", color="#8e44ad", linewidth=2.2, label="Dynamic TOU Tariff ($/kWh)")
    ax.fill_between(profile_df["time_hours"], 0, profile_df["electricity_price_per_kwh"], step="post", color="#8e44ad", alpha=0.2)

    # Annotate TOU pricing regimes
    ax.axvspan(0, 6, color="#2ecc71", alpha=0.08, label="Off-Peak ($0.08)")
    ax.axvspan(6, 16, color="#f1c40f", alpha=0.08, label="Mid-Peak ($0.15)")
    ax.axvspan(16, 21, color="#e74c3c", alpha=0.10, label="On-Peak ($0.32)")
    ax.axvspan(21, 23, color="#f1c40f", alpha=0.08)
    ax.axvspan(23, 24, color="#2ecc71", alpha=0.08)

    ax.set_title("24-Hour Time-of-Use (TOU) Electricity Pricing Tariff", fontweight="bold", pad=12)
    ax.set_xlabel("Time of Day (Hours)")
    ax.set_ylabel("Tariff Rate ($/kWh)")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 0.40)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved: {save_path.name}")
    return save_path


def plot_ev_arrivals(evs_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Graph 4: EV arrival and departure visualization."""
    if save_path is None:
        save_path = PLOTS_DIR / "04_ev_arrivals_departures.png"

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

    y_pos = np.arange(len(evs_df))
    durations = evs_df["departure_time_hours"] - evs_df["arrival_time_hours"]

    # Bar horizontal representing dwell span
    bars = ax.barh(
        y_pos,
        durations,
        left=evs_df["arrival_time_hours"],
        height=0.45,
        color="#1abc9c",
        edgecolor="#16a085",
        linewidth=1.5,
        alpha=0.85,
        label="Dwell Connection Window",
    )

    # Annotate initial and target SOC on each bar
    for idx, row in evs_df.iterrows():
        mid_x = row["arrival_time_hours"] + (row["departure_time_hours"] - row["arrival_time_hours"]) / 2.0
        ax.text(
            mid_x,
            idx,
            f"EV {int(row['ev_id'])}: {row['initial_soc']:.0f}% -> {row['target_soc']:.0f}%",
            ha="center",
            va="center",
            color="black",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Port Slot {int(row['ev_id'])}" for _, row in evs_df.iterrows()])
    ax.set_xlabel("Time of Day (Hours)")
    ax.set_title("24-Hour EV Arrival, Departure, and Dwell Schedule", fontweight="bold", pad=12)
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved: {save_path.name}")
    return save_path


def generate_all_environment_plots(profile_df: pd.DataFrame, evs_df: pd.DataFrame) -> None:
    """Convenience function generating all 4 environmental graphs."""
    plot_solar_generation(profile_df)
    plot_grid_demand(profile_df)
    plot_electricity_price(profile_df)
    plot_ev_arrivals(evs_df)
