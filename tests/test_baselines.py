"""Unit tests for baseline algorithms (Immediate Charging and Solar-First Charging)."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import NUM_TIME_STEPS, EV_TARGET_SOC
from environment.ev_model import EV
from environment.ev_environment import EVChargingEnv
from baselines.immediate_charging import ImmediateChargingPolicy
from baselines.solar_first import SolarFirstChargingPolicy


def test_immediate_charging_policy():
    """Immediate charging policy must request Action 2 (7 kW) for all eligible connected EVs."""
    env = EVChargingEnv()
    profile_df = pd.DataFrame({
        "step": np.arange(NUM_TIME_STEPS),
        "solar_generation_kw": np.zeros(NUM_TIME_STEPS),
        "grid_demand_kw": np.full(NUM_TIME_STEPS, 20.0),
        "electricity_price_per_kwh": np.full(NUM_TIME_STEPS, 0.15),
    })

    ev1 = EV(ev_id=1, arrival_step=0, departure_step=20, initial_soc=30.0)
    ev2 = EV(ev_id=2, arrival_step=0, departure_step=20, initial_soc=50.0)
    env.load_episode_data(profile_df, [ev1, ev2])
    env.reset()

    policy = ImmediateChargingPolicy()
    actions = policy.get_actions(env)

    assert actions[1] == 2
    assert actions[2] == 2


def test_solar_first_policy_behavior():
    """Solar-first policy must utilize solar first and only draw grid power when urgency is high."""
    env = EVChargingEnv()
    profile_df = pd.DataFrame({
        "step": np.arange(NUM_TIME_STEPS),
        "solar_generation_kw": np.zeros(NUM_TIME_STEPS),  # Zero solar
        "grid_demand_kw": np.full(NUM_TIME_STEPS, 20.0),
        "electricity_price_per_kwh": np.full(NUM_TIME_STEPS, 0.30),  # Expensive peak
    })

    # EV with low urgency (ample time: arrival 0, departure 50, initial SOC 70%)
    ev_relaxed = EV(ev_id=1, arrival_step=0, departure_step=50, initial_soc=70.0)
    # EV with high urgency (arrival 0, departure 3, initial SOC 20%)
    ev_urgent = EV(ev_id=2, arrival_step=0, departure_step=3, initial_soc=20.0)

    env.load_episode_data(profile_df, [ev_relaxed, ev_urgent])
    env.reset()

    policy = SolarFirstChargingPolicy()
    actions = policy.get_actions(env)

    # Relaxed EV should wait (Action 0) because solar is zero and price is expensive
    assert actions[1] == 0
    # Urgent EV must charge (Action 2) to prevent departure deficit
    assert actions[2] == 2

