"""Unit tests for the EV Charging Station Environment and physical models."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    NUM_TIME_STEPS,
    TOTAL_STATION_POWER_LIMIT_KW,
    EV_MAX_CHARGING_POWER_KW,
    EV_BATTERY_CAPACITY_KWH,
    EV_CHARGING_EFFICIENCY,
    EV_TARGET_SOC,
    EV_MAX_SOC,
)
from environment.solar_model import SolarModel
from environment.grid_model import GridModel
from environment.ev_model import EV
from environment.reward_function import RewardFunction
from environment.ev_environment import EVChargingEnv


def test_solar_model_properties():
    """Solar generation must be non-negative, zero at night, and peak during midday."""
    rng = np.random.default_rng(42)
    solar_model = SolarModel(rated_capacity_kw=18.0)
    profile = solar_model.generate_profile(rng=rng)

    assert len(profile) == NUM_TIME_STEPS
    assert np.all(profile >= 0.0), "Solar generation must be strictly non-negative"
    # Midnight to 5 AM should be zero
    assert np.all(profile[:20] == 0.0), "Solar power must be zero during night"
    # Midday should show positive solar generation
    assert np.max(profile[40:60]) > 5.0, "Solar generation must peak around midday"


def test_grid_model_properties():
    """Grid demand and electricity prices must be non-negative and physically bounded."""
    rng = np.random.default_rng(42)
    grid_model = GridModel()
    demand, price = grid_model.generate_grid_scenario(rng=rng)

    assert len(demand) == NUM_TIME_STEPS
    assert len(price) == NUM_TIME_STEPS
    assert np.all(demand >= 0.0), "Grid demand must be non-negative"
    assert np.all(price > 0.0), "Electricity price must be strictly positive"
    # Evening peak demand should exceed midday demand
    assert np.max(demand[64:84]) > np.mean(demand[:24])


def test_ev_physical_charging_laws():
    """Verifies that charging follows conservation of energy and exact SOC update formula."""
    ev = EV(
        ev_id=1,
        arrival_step=10,
        departure_step=30,
        initial_soc=20.0,
        target_soc=80.0,
        battery_capacity_kwh=40.0,
        max_charging_power_kw=7.0,
        charging_efficiency=0.90,
    )

    assert ev.current_soc == 20.0
    assert not ev.is_present(9)
    assert ev.is_present(10)
    assert ev.is_present(29)
    assert not ev.is_present(30)

    # 15 min at 7 kW: Energy = 7 * 0.25 = 1.75 kWh. Stored = 1.75 * 0.90 = 1.575 kWh
    # Delta SOC = (1.575 / 40.0) * 100 = 3.9375%
    rec = ev.charge(7.0, time_interval_h=0.25)
    expected_delta_soc = (7.0 * 0.25 * 0.90 / 40.0) * 100.0
    assert np.isclose(rec["delta_soc"], expected_delta_soc, atol=1e-4)
    assert np.isclose(ev.current_soc, 20.0 + expected_delta_soc, atol=1e-4)


def test_ev_soc_hard_cap_at_100():
    """SOC must never exceed 100% even if excessive charging is requested."""
    ev = EV(
        ev_id=2,
        arrival_step=0,
        departure_step=50,
        initial_soc=98.0,
        target_soc=80.0,
    )

    # Charge at 7 kW (which would normally add ~3.9%)
    rec = ev.charge(7.0, time_interval_h=0.25)
    assert ev.current_soc <= 100.0
    assert np.isclose(ev.current_soc, 100.0, atol=1e-4)

    # Subsequent charge should add 0
    rec2 = ev.charge(7.0, time_interval_h=0.25)
    assert rec2["delta_soc"] == 0.0
    assert ev.current_soc == 100.0


def test_reward_function_components():
    """Reward function should properly balance cost, solar, peak and unmet penalty."""
    rf = RewardFunction(alpha_cost=1.0, beta_peak=0.3, gamma_solar=1.2, delta_unmet=4.0)

    # Zero cost, zero solar, low load
    res1 = rf.calculate_step_reward(
        grid_electricity_cost=0.0,
        solar_energy_used_kwh=0.0,
        total_grid_power_kw=20.0,
        unmet_energy_kwh=0.0,
    )
    assert res1["total_reward"] == 0.0

    # Solar usage of 2 kWh with zero grid cost
    res2 = rf.calculate_step_reward(
        grid_electricity_cost=0.0,
        solar_energy_used_kwh=2.0,
        total_grid_power_kw=20.0,
        unmet_energy_kwh=0.0,
    )
    assert np.isclose(res2["total_reward"], 2.4, atol=1e-3)

    # Unmet departure energy penalty of 5 kWh deficit
    res3 = rf.calculate_step_reward(
        grid_electricity_cost=0.50,
        solar_energy_used_kwh=0.0,
        total_grid_power_kw=40.0,  # 5 kW above threshold
        unmet_energy_kwh=5.0,
    )
    # cost_penalty = 0.50, peak_penalty = 0.3 * (5/10) = 0.15, unmet = 4 * 5 = 20.0
    expected_reward = -0.50 - 0.15 - 20.0
    assert np.isclose(res3["total_reward"], expected_reward, atol=1e-3)

