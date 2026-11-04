"""Comparative visualization module for benchmarking RL against baselines and experimental analysis.
Generates Publication-Quality plots for:
8. Electricity cost comparison across all three strategies
9. Solar energy utilization comparison
10. Peak grid demand comparison
11. Grid energy consumption comparison
12. Departure SOC satisfaction comparison
13. Learning rate experiment comparison
14. Solar availability sensitivity analysis
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

from config import PLOTS_DIR


# Consistent academic strategy color palette
STRATEGY_COLORS = {
    "Immediate Charging": "#e74c3c",       # Crimson / uncontrolled
    "Solar-First Charging": "#f39c12",     # Amber / renewable heuristic
    "RL (Q-Learning)": "#27ae60",          # Emerald / intelligent RL
}


def _create_strategy_bar_plot(
    detailed_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    save_filename: str,
    higher_is_better: bool = False,
    unit_fmt: str = "{:.2f}",
) -> Path:
    """Helper method for creating polished publication-style 3-strategy comparison bar plots with error bars."""
    save_path = PLOTS_DIR / save_filename

    strategies = ["Immediate Charging", "Solar-First Charging", "RL (Q-Learning)"]
    means = []
    stds = []
    colors = []

    for s in strategies:
        data = detailed_df[detailed_df["strategy"] == s][metric_col]
        means.append(float(data.mean()))
        stds.append(float(data.std()))
        colors.append(STRATEGY_COLORS.get(s, "#34495e"))

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    x = np.arange(len(strategies))
    width = 0.55

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        width=width,
        color=colors,
        alpha=0.88,
        edgecolor="#2c3e50",
        linewidth=1.2,
        capsize=6,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
    )

    # Value annotations on top of bars
    for bar, m, s_val in zip(bars, means, stds):
        offset = max(0.5, s_val * 0.4)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + s_val + offset,
            f"{unit_fmt.format(m)} \u00b1 {s_val:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=14)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    y_top = max([m + s for m, s in zip(means, stds)]) * 1.25
    ax.set_ylim(0, max(1.0, y_top))

    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved: {save_path.name}")
    return save_path


def plot_cost_comparison(detailed_df: pd.DataFrame) -> Path:
    """Graph 8: Electricity cost comparison: all three strategies."""
    return _create_strategy_bar_plot(
        detailed_df=detailed_df,
        metric_col="grid_cost_usd",
        title="Total Grid Electricity Cost Comparison (20 Test Episodes)",
        ylabel="Total Cost ($)",
        save_filename="08_cost_comparison.png",
        unit_fmt="${:.2f}",
    )


def plot_solar_utilization_comparison(detailed_df: pd.DataFrame) -> Path:
    """Graph 9: Solar energy utilization comparison."""
    return _create_strategy_bar_plot(
        detailed_df=detailed_df,
        metric_col="solar_energy_kwh",
        title="Direct Solar PV Energy Utilization for EV Charging",
        ylabel="Solar Energy Utilized (kWh)",
        save_filename="09_solar_utilization_comparison.png",
        unit_fmt="{:.2f} kWh",
    )


def plot_peak_demand_comparison(detailed_df: pd.DataFrame) -> Path:
    """Graph 10: Peak grid demand comparison."""
    return _create_strategy_bar_plot(
        detailed_df=detailed_df,
        metric_col="peak_grid_demand_kw",
        title="Peak Grid Power Demand Comparison (Peak Load Shaving)",
        ylabel="Peak Grid Demand (kW)",
        save_filename="10_peak_grid_demand_comparison.png",
        unit_fmt="{:.2f} kW",
    )


def plot_grid_energy_consumption_comparison(detailed_df: pd.DataFrame) -> Path:
    """Graph 11: Grid energy consumption comparison."""
    return _create_strategy_bar_plot(
        detailed_df=detailed_df,
        metric_col="grid_energy_kwh",
        title="Total Grid Energy Consumed for EV Charging",
        ylabel="Grid Energy Consumed (kWh)",
        save_filename="11_grid_energy_consumption_comparison.png",
        unit_fmt="{:.2f} kWh",
    )


def plot_departure_soc_satisfaction_comparison(detailed_df: pd.DataFrame) -> Path:
    """Graph 12: Departure SOC satisfaction comparison."""
    save_path = PLOTS_DIR / "12_departure_soc_satisfaction_comparison.png"

    strategies = ["Immediate Charging", "Solar-First Charging", "RL (Q-Learning)"]
    soc_means = [float(detailed_df[detailed_df["strategy"] == s]["avg_departure_soc_pct"].mean()) for s in strategies]
    soc_stds = [float(detailed_df[detailed_df["strategy"] == s]["avg_departure_soc_pct"].std()) for s in strategies]

    satisfaction_pcts = [float(detailed_df[detailed_df["strategy"] == s]["target_soc_achieved_pct"].mean()) for s in strategies]

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(len(strategies))
    width = 0.35

    rects1 = ax1.bar(x - width/2, soc_means, width, yerr=soc_stds, label="Avg Departure SOC (%)", color="#3498db", capsize=5, edgecolor="#2980b9")
    ax1.set_ylabel("Average Departure SOC (%)", color="#2980b9")
    ax1.tick_params(axis="y", labelcolor="#2980b9")
    ax1.set_ylim(0, 110)
    ax1.axhline(80.0, color="#c0392b", linestyle=":", linewidth=1.8, label="Target SOC (80%)")

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, satisfaction_pcts, width, label="% Achieving Target SOC", color="#2ecc71", alpha=0.85, edgecolor="#27ae60")
    ax2.set_ylabel("EVs Achieving Target SOC (%)", color="#27ae60")
    ax2.tick_params(axis="y", labelcolor="#27ae60")
    ax2.set_ylim(0, 115)

    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontweight="bold")
    ax1.set_title("EV Departure SOC & Fleet Target Satisfaction Comparison", fontweight="bold", pad=12)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", framealpha=0.9)

    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved: {save_path.name}")
    return save_path


def plot_learning_rate_comparison(lr_history_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Graph 13: Learning rate experiment comparison (Convergence & Reward trajectories)."""
    if save_path is None:
        save_path = PLOTS_DIR / "13_learning_rate_experiment_comparison.png"

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    lr_colors = {0.01: "#e67e22", 0.1: "#2980b9", 0.5: "#9b59b6"}

    for lr in sorted(lr_history_df["learning_rate"].unique()):
        sub_df = lr_history_df[lr_history_df["learning_rate"] == lr].sort_values("episode")
        episodes = sub_df["episode"]
        rewards = sub_df["total_episode_reward"]
        rolling_rew = rewards.rolling(window=50, min_periods=1).mean()

        color = lr_colors.get(lr, "#34495e")
        ax.plot(episodes, rolling_rew, label=f"Learning Rate alpha = {lr}", color=color, linewidth=2.2)

    ax.set_title("Learning Rate Convergence Analysis (Tabular Q-Learning)", fontweight="bold", pad=12)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("50-Episode Moving Average Reward")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)

    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved: {save_path.name}")
    return save_path


def plot_solar_sensitivity(sensitivity_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Graph 14: Solar availability sensitivity analysis."""
    if save_path is None:
        save_path = PLOTS_DIR / "14_solar_sensitivity_analysis.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    strategies = ["Immediate Charging", "Solar-First Charging", "RL (Q-Learning)"]
    x = np.arange(len(strategies))
    width = 0.35

    # Subplot 1: Grid Electricity Cost impact
    nom_costs = [
        float(sensitivity_df[(sensitivity_df["solar_condition"] == "Nominal Solar (100%)") & (sensitivity_df["strategy"] == s)]["avg_grid_cost_usd"].iloc[0])
        for s in strategies
    ]
    red_costs = [
        float(sensitivity_df[(sensitivity_df["solar_condition"] == "Reduced Solar (70%)") & (sensitivity_df["strategy"] == s)]["avg_grid_cost_usd"].iloc[0])
        for s in strategies
    ]

    ax1.bar(x - width/2, nom_costs, width, label="Nominal Solar (100%)", color="#2ecc71", edgecolor="#27ae60")
    ax1.bar(x + width/2, red_costs, width, label="Reduced Solar (70%)", color="#e67e22", edgecolor="#d35400")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=15, ha="right", fontweight="bold")
    ax1.set_ylabel("Avg Grid Electricity Cost ($)")
    ax1.set_title("(A) Electricity Cost Impact", fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # Subplot 2: Solar Utilization percentage impact
    nom_solar = [
        float(sensitivity_df[(sensitivity_df["solar_condition"] == "Nominal Solar (100%)") & (sensitivity_df["strategy"] == s)]["solar_contribution_pct"].iloc[0])
        for s in strategies
    ]
    red_solar = [
        float(sensitivity_df[(sensitivity_df["solar_condition"] == "Reduced Solar (70%)") & (sensitivity_df["strategy"] == s)]["solar_contribution_pct"].iloc[0])
        for s in strategies
    ]

    ax2.bar(x - width/2, nom_solar, width, label="Nominal Solar (100%)", color="#2ecc71", edgecolor="#27ae60")
    ax2.bar(x + width/2, red_solar, width, label="Reduced Solar (70%)", color="#e67e22", edgecolor="#d35400")
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=15, ha="right", fontweight="bold")
    ax2.set_ylabel("Solar Energy Contribution (%)")
    ax2.set_title("(B) Solar Energy Contribution Impact", fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.suptitle("Environmental Sensitivity Analysis: Impact of 30% Solar Reduction", fontweight="bold", y=1.02)
    fig.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {save_path.name}")
    return save_path

