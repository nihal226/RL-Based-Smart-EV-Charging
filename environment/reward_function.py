"""Reward function module for the Reinforcement Learning EV Charging Environment.
Implements a mathematically justified, multi-objective scalar reward balancing:
1. Electricity purchasing cost minimization.
2. Peak grid demand reduction.
3. Renewable solar self-consumption maximization.
4. EV target departure SOC guarantee (satisfaction constraint).
"""

from typing import Dict, Any
from config import (
    ALPHA_COST,
    BETA_PEAK,
    GAMMA_SOLAR,
    DELTA_UNMET,
    INVALID_ACTION_PENALTY,
    PEAK_GRID_THRESHOLD_KW,
)


class RewardFunction:
    """Calculates weighted multi-objective rewards for EV charging scheduling decisions."""

    def __init__(
        self,
        alpha_cost: float = ALPHA_COST,
        beta_peak: float = BETA_PEAK,
        gamma_solar: float = GAMMA_SOLAR,
        delta_unmet: float = DELTA_UNMET,
        invalid_action_penalty: float = INVALID_ACTION_PENALTY,
    ):
        self.alpha_cost = alpha_cost
        self.beta_peak = beta_peak
        self.gamma_solar = gamma_solar
        self.delta_unmet = delta_unmet
        self.invalid_action_penalty = invalid_action_penalty

    def calculate_step_reward(
        self,
        grid_electricity_cost: float,
        solar_energy_used_kwh: float,
        total_grid_power_kw: float,
        unmet_energy_kwh: float = 0.0,
        is_invalid_action: bool = False,
    ) -> Dict[str, float]:
        """Calculates the scalar reward and individual components for a single time step.

        Parameters:
        -----------
        grid_electricity_cost : float
            Cost of grid electricity drawn during this step ($).
        solar_energy_used_kwh : float
            Quantity of clean solar energy delivered to EVs (kWh).
        total_grid_power_kw : float
            Aggregate grid power including background demand and EV charging (kW).
        unmet_energy_kwh : float, optional
            Total unmet energy deficit for EVs departing in this step (kWh).
        is_invalid_action : bool, optional
            Flag indicating if agent selected an impermissible action.

        Returns:
        --------
        Dict[str, float]: Contains total reward and breakdown of individual components.
        """
        # 1. Grid electricity purchasing cost penalty
        cost_penalty = self.alpha_cost * grid_electricity_cost

        # 2. Peak grid demand penalty (applies when total load exceeds substation capacity threshold)
        peak_excess_kw = max(0.0, total_grid_power_kw - PEAK_GRID_THRESHOLD_KW)
        peak_penalty = self.beta_peak * (peak_excess_kw / 10.0)

        # 3. Solar energy utilization bonus
        solar_reward = self.gamma_solar * solar_energy_used_kwh

        # 4. Unmet departure energy penalty (heavily weighted to prevent EV starvation)
        unmet_penalty = self.delta_unmet * unmet_energy_kwh

        # 5. Invalid action penalty
        invalid_penalty = self.invalid_action_penalty if is_invalid_action else 0.0

        # Aggregate weighted reward
        total_reward = (
            - cost_penalty
            - peak_penalty
            + solar_reward
            - unmet_penalty
            - invalid_penalty
        )

        return {
            "total_reward": round(total_reward, 4),
            "cost_penalty": round(cost_penalty, 4),
            "peak_penalty": round(peak_penalty, 4),
            "solar_reward": round(solar_reward, 4),
            "unmet_penalty": round(unmet_penalty, 4),
            "invalid_penalty": round(invalid_penalty, 4),
        }

