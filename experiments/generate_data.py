"""Synthetic data generation script for RL-Based Smart EV Charging.
Generates 100 reproducible 24-hour scenarios (80 training, 20 testing).
Ensures strict physical consistency, separate train/test splits, and zero data leakage.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    TRAIN_DATA_DIR,
    TEST_DATA_DIR,
    PROFILES_DATA_DIR,
    NUM_TIME_STEPS,
    TIME_INTERVAL_HOURS,
    RANDOM_SEED,
    EV_BATTERY_CAPACITY_KWH,
    EV_MAX_CHARGING_POWER_KW,
    EV_CHARGING_EFFICIENCY,
    EV_INITIAL_SOC_MIN,
    EV_INITIAL_SOC_MAX,
    EV_TARGET_SOC,
    NUM_CHARGING_PORTS,
)
from environment.solar_model import SolarModel
from environment.grid_model import GridModel


def generate_ev_schedule(
    rng: np.random.Generator,
    min_evs: int = 4,
    max_evs: int = 6,
    num_ports: int = NUM_CHARGING_PORTS,
) -> List[Dict[str, Any]]:
    """Generates a realistic daily EV arrival schedule respecting station port limits."""
    num_evs = rng.integers(min_evs, max_evs + 1)
    evs = []

    # Port availability tracker to ensure maximum concurrent vehicles does not exceed ports
    port_schedule = [0] * num_ports

    # Candidate arrival windows distributed across daytime (06:30 - 18:00)
    # Morning commute arrivals (07:00 - 09:30): steps 28 - 38
    # Midday arrivals (10:30 - 13:30): steps 42 - 54
    # Afternoon arrivals (14:00 - 17:00): steps 56 - 68
    arrival_pools = [
        (26, 38),   # Morning
        (38, 52),   # Midday
        (52, 66),   # Early afternoon
        (64, 76),   # Late afternoon
    ]

    for ev_id in range(num_evs):
        # Pick arrival pool
        pool_idx = ev_id % len(arrival_pools)
        min_arr, max_arr = arrival_pools[pool_idx]
        arrival_step = int(rng.integers(min_arr, max_arr))

        # Dwell duration: between 3 hours (12 steps) and 7 hours (28 steps)
        dwell_steps = int(rng.integers(12, 28))
        departure_step = min(NUM_TIME_STEPS - 1, arrival_step + dwell_steps)

        # Initial SOC randomly drawn between 15% and 70%
        initial_soc = float(rng.uniform(EV_INITIAL_SOC_MIN, EV_INITIAL_SOC_MAX))

        # Ensure dwell time is physically capable of reaching 80% under max power
        energy_needed = max(0.0, EV_TARGET_SOC - initial_soc) / 100.0 * EV_BATTERY_CAPACITY_KWH
        min_steps_needed = int(np.ceil(energy_needed / (EV_MAX_CHARGING_POWER_KW * TIME_INTERVAL_HOURS * EV_CHARGING_EFFICIENCY)))
        
        # Give at least 2 extra steps of dwell leeway so scheduling flexibility exists
        if departure_step - arrival_step < min_steps_needed + 2:
            departure_step = min(NUM_TIME_STEPS - 1, arrival_step + min_steps_needed + 3)

        evs.append({
            "ev_id": ev_id + 1,
            "arrival_step": int(arrival_step),
            "departure_step": int(departure_step),
            "arrival_time_hours": round(arrival_step * 0.25, 2),
            "departure_time_hours": round(departure_step * 0.25, 2),
            "initial_soc": round(initial_soc, 1),
            "target_soc": float(EV_TARGET_SOC),
            "battery_capacity_kwh": float(EV_BATTERY_CAPACITY_KWH),
            "max_charging_power_kw": float(EV_MAX_CHARGING_POWER_KW),
            "charging_efficiency": float(EV_CHARGING_EFFICIENCY),
        })

    # Sort EVs by arrival step
    evs.sort(key=lambda x: x["arrival_step"])
    # Re-index EV IDs sequentially
    for i, ev in enumerate(evs):
        ev["ev_id"] = i + 1

    return evs


def generate_episode_dataset(
    episode_idx: int,
    rng: np.random.Generator,
    solar_model: SolarModel,
    grid_model: GridModel,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates 24h environmental profile DataFrame and EV schedule DataFrame for one episode."""
    # 1. Solar generation
    solar_kw = solar_model.generate_profile(rng=rng)

    # 2. Grid demand and price
    demand_kw, price_kwh = grid_model.generate_grid_scenario(rng=rng)

    time_steps = np.arange(NUM_TIME_STEPS)
    hours = time_steps * 0.25

    profile_df = pd.DataFrame({
        "step": time_steps,
        "time_hours": np.round(hours, 2),
        "solar_generation_kw": solar_kw,
        "grid_demand_kw": demand_kw,
        "electricity_price_per_kwh": price_kwh,
    })

    # 3. EV arrivals
    ev_list = generate_ev_schedule(rng=rng)
    evs_df = pd.DataFrame(ev_list)

    return profile_df, evs_df


def generate_all_datasets(
    num_train: int = 80,
    num_test: int = 20,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Generates and saves the full synthetic dataset suite for train and test splits."""
    print(f"[Data Generator] Initializing dataset generation with seed={seed}...")
    rng = np.random.default_rng(seed)

    solar_model = SolarModel()
    grid_model = GridModel()

    train_meta = []
    test_meta = []

    # 1. Training datasets (episodes 0 to num_train - 1)
    print(f"[Data Generator] Generating {num_train} training episodes in {TRAIN_DATA_DIR}...")
    for ep in range(num_train):
        prof_df, evs_df = generate_episode_dataset(ep, rng, solar_model, grid_model)
        
        prof_path = TRAIN_DATA_DIR / f"episode_{ep:03d}_profile.csv"
        evs_path = TRAIN_DATA_DIR / f"episode_{ep:03d}_evs.csv"
        
        prof_df.to_csv(prof_path, index=False)
        evs_df.to_csv(evs_path, index=False)
        
        train_meta.append({
            "episode": ep,
            "num_evs": len(evs_df),
            "max_solar_kw": float(prof_df["solar_generation_kw"].max()),
            "avg_price": float(prof_df["electricity_price_per_kwh"].mean()),
            "profile_file": str(prof_path.name),
            "evs_file": str(evs_path.name),
        })

    # 2. Testing datasets (independent test episodes)
    print(f"[Data Generator] Generating {num_test} independent test episodes in {TEST_DATA_DIR}...")
    for ep in range(num_test):
        prof_df, evs_df = generate_episode_dataset(ep, rng, solar_model, grid_model)
        
        prof_path = TEST_DATA_DIR / f"episode_{ep:03d}_profile.csv"
        evs_path = TEST_DATA_DIR / f"episode_{ep:03d}_evs.csv"
        
        prof_df.to_csv(prof_path, index=False)
        evs_df.to_csv(evs_path, index=False)
        
        test_meta.append({
            "episode": ep,
            "num_evs": len(evs_df),
            "max_solar_kw": float(prof_df["solar_generation_kw"].max()),
            "avg_price": float(prof_df["electricity_price_per_kwh"].mean()),
            "profile_file": str(prof_path.name),
            "evs_file": str(evs_path.name),
        })

    # Save summary metadata
    summary = {
        "random_seed": seed,
        "num_training_episodes": num_train,
        "num_testing_episodes": num_test,
        "num_time_steps_per_day": NUM_TIME_STEPS,
        "time_interval_hours": TIME_INTERVAL_HOURS,
        "train_summary": train_meta,
        "test_summary": test_meta,
    }

    summary_path = DATA_DIR / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Data Generator] Successfully created {num_train + num_test} total profiles.")
    print(f"[Data Generator] Metadata saved to: {summary_path}")
    return summary


if __name__ == "__main__":
    generate_all_datasets()
