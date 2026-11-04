"""Unit tests for energy constraints and station limits."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    TOTAL_STATION_POWER_LIMIT_KW,
    EV_MAX_CHARGING_POWER_KW,
    EV_TARGET_SOC,
    NUM_TIME_STEPS,
)
from environment.ev_model import EV
from environment.ev_environment import EVChargingEnv


def test_station_capacity_constraint():
    """Environment must enforce the 21 kW total station power limit across multiple EVs."""
    env = EVChargingEnv(station_power_limit_kw=TOTAL_STATION_POWER_LIMIT_KW)

    # Synthetic profile with high demand
    profile_df = pd.DataFrame({
        "step": np.arange(NUM_TIME_STEPS),
        "solar_generation_kw": np.zeros(NUM_TIME_STEPS),
        "grid_demand_kw": np.full(NUM_TIME_STEPS, 20.0),
        "electricity_price_per_kwh": np.full(NUM_TIME_STEPS, 0.15),
    })

    # Create 4 EVs all present at step 10 with low SOC
    evs = [
        EV(ev_id=i, arrival_step=10, departure_step=30, initial_soc=20.0, target_soc=80.0)
        for i in range(1, 5)
    ]

    env.load_episode_data(profile_df, evs)
    env.reset()

    # Advance environment to step 10
    for _ in range(10):
        env.step({})

    assert env.current_step == 10
    # All 4 EVs attempt to charge at Action 2 (7 kW each) -> total requested = 28 kW!
    actions = {1: 2, 2: 2, 3: 2, 4: 2}
    step_info, _, _, _ = env.step(actions)

    # Enforce station capacity limit of 21 kW
    assert step_info["total_ev_power_kw"] <= TOTAL_STATION_POWER_LIMIT_KW
    assert np.isclose(step_info["total_ev_power_kw"], TOTAL_STATION_POWER_LIMIT_KW, atol=1e-3)


def test_cannot_charge_before_arrival_or_after_departure():
    """EVs must not receive charging power prior to arrival or post departure."""
    env = EVChargingEnv()

    profile_df = pd.DataFrame({
        "step": np.arange(NUM_TIME_STEPS),
        "solar_generation_kw": np.full(NUM_TIME_STEPS, 10.0),
        "grid_demand_kw": np.full(NUM_TIME_STEPS, 20.0),
        "electricity_price_per_kwh": np.full(NUM_TIME_STEPS, 0.15),
    })

    ev = EV(ev_id=1, arrival_step=15, departure_step=25, initial_soc=30.0)
    env.load_episode_data(profile_df, [ev])
    env.reset()

    # Step 5: EV has not arrived yet
    for _ in range(5):
        env.step({})

    assert env.current_step == 5
    assert not ev.is_present(5)

    # Attempt to charge at step 5
    step_info, _, _, _ = env.step({1: 2})
    assert step_info["total_ev_power_kw"] == 0.0
    assert ev.current_soc == 30.0

    # Fast forward past departure step (step 25)
    while env.current_step < 25:
        env.step({})

    assert env.current_step == 25
    assert not ev.is_present(25)

    # Attempt to charge after departure
    step_info, _, _, _ = env.step({1: 2})
    assert step_info["total_ev_power_kw"] == 0.0


def test_target_soc_stop_charging():
    """Once target SOC is achieved, the EV should not be eligible for further charging."""
    ev = EV(ev_id=1, arrival_step=0, departure_step=50, initial_soc=79.5, target_soc=80.0)
    assert ev.is_eligible_to_charge(0)

    # Charge slightly to reach >= 80.0
    ev.charge(3.5, time_interval_h=0.25)
    assert ev.current_soc >= 80.0
    assert not ev.is_eligible_to_charge(1)

