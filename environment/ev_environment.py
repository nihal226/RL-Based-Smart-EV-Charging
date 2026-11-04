"""EV Charging Station Environment.
Simulates a multi-port 24-hour EV charging station with on-site solar generation,
dynamic grid demand, Time-of-Use pricing, and physical battery dynamics.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd

from config import (
    NUM_TIME_STEPS,
    TIME_INTERVAL_HOURS,
    NUM_CHARGING_PORTS,
    TOTAL_STATION_POWER_LIMIT_KW,
    EV_MAX_CHARGING_POWER_KW,
    EV_TARGET_SOC,
    EV_MAX_SOC,
    ACTION_POWER_MAP,
    ACTIONS,
    TIME_BINS,
    SOC_BINS,
    SOLAR_BINS,
    GRID_BINS,
    PRICE_BINS,
    URGENCY_BINS,
    OCCUPANCY_BINS,
)
from environment.ev_model import EV
from environment.reward_function import RewardFunction


class EVChargingEnv:
    """Discrete-time 24-hour simulation environment for smart EV charging station."""

    def __init__(
        self,
        reward_function: Optional[RewardFunction] = None,
        num_ports: int = NUM_CHARGING_PORTS,
        station_power_limit_kw: float = TOTAL_STATION_POWER_LIMIT_KW,
    ):
        self.num_ports = num_ports
        self.station_power_limit_kw = station_power_limit_kw
        self.reward_fn = reward_function if reward_function is not None else RewardFunction()

        # Episode data
        self.current_step: int = 0
        self.profile_df: Optional[pd.DataFrame] = None
        self.evs: List[EV] = []
        self.active_evs: List[EV] = []

        # History tracking across 96 steps
        self.history: Dict[str, List[float]] = {}
        self.completed_evs: List[Dict[str, Any]] = []

    def load_episode_from_csv(self, profile_path: Path, evs_path: Path) -> None:
        """Loads environment profile and EV schedule from CSV files."""
        self.profile_df = pd.read_csv(profile_path)
        evs_df = pd.read_csv(evs_path)

        self.evs = []
        for _, row in evs_df.iterrows():
            ev = EV(
                ev_id=int(row["ev_id"]),
                arrival_step=int(row["arrival_step"]),
                departure_step=int(row["departure_step"]),
                initial_soc=float(row["initial_soc"]),
                target_soc=float(row.get("target_soc", EV_TARGET_SOC)),
                battery_capacity_kwh=float(row.get("battery_capacity_kwh", 40.0)),
                max_charging_power_kw=float(row.get("max_charging_power_kw", EV_MAX_CHARGING_POWER_KW)),
                charging_efficiency=float(row.get("charging_efficiency", 0.90)),
            )
            self.evs.append(ev)

    def load_episode_data(self, profile_df: pd.DataFrame, evs_list: List[EV]) -> None:
        """Loads environment profile DataFrame and EV objects directly."""
        self.profile_df = profile_df.copy()
        # Create fresh copies of EV objects
        self.evs = [
            EV(
                ev_id=ev.ev_id,
                arrival_step=ev.arrival_step,
                departure_step=ev.departure_step,
                initial_soc=ev.initial_soc,
                target_soc=ev.target_soc,
                battery_capacity_kwh=ev.battery_capacity_kwh,
                max_charging_power_kw=ev.max_charging_power_kw,
                charging_efficiency=ev.charging_efficiency,
            )
            for ev in evs_list
        ]

    def reset(self) -> None:
        """Resets the environment to step 0 for a new simulation run."""
        self.current_step = 0
        self.active_evs = []
        self.completed_evs = []

        # Reset all EV states
        for ev in self.evs:
            ev.current_soc = ev.initial_soc
            ev.departure_soc = ev.initial_soc
            ev.total_energy_charged_kwh = 0.0
            ev.total_energy_drawn_kwh = 0.0
            ev.is_completed = False

        self.history = {
            "step": [],
            "time_hours": [],
            "solar_generation_kw": [],
            "base_grid_demand_kw": [],
            "total_grid_demand_kw": [],
            "electricity_price": [],
            "total_ev_power_kw": [],
            "solar_power_used_kw": [],
            "grid_power_for_ev_kw": [],
            "step_electricity_cost": [],
            "step_reward": [],
            "active_ev_count": [],
        }

    # ==========================================
    # State Space Discretization
    # ==========================================
    def discretize_state(self, ev: Optional[EV]) -> Tuple[int, int, int, int, int, int, int]:
        """Discretizes the continuous environment and EV state into 7 discrete bins.

        Features:
        1. Time category (0: Night, 1: Morning, 2: Midday, 3: Evening)
        2. SOC category (0: Low <50%, 1: Med 50-75%, 2: Target/High >=75%)
        3. Solar category (0: Low <2kW, 1: Med 2-8kW, 2: High >=8kW)
        4. Grid demand category (0: Normal <35kW, 1: Peak >=35kW)
        5. Price category (0: Low <0.12, 1: Med 0.12-0.22, 2: High >=0.22)
        6. Urgency category (0: Low <0.40, 1: Med 0.40-0.75, 2: High >=0.75)
        7. Station occupancy category (0: Low <=1 EV, 1: High >=2 EVs)

        Returns:
        --------
        Tuple[int, int, int, int, int, int, int]: 7-element discrete state tuple.
        """
        # 1. Time bin
        t = min(NUM_TIME_STEPS - 1, self.current_step)
        if t < TIME_BINS[0]:
            time_cat = 0
        elif t < TIME_BINS[1]:
            time_cat = 1
        elif t < TIME_BINS[2]:
            time_cat = 2
        else:
            time_cat = 3

        # Current environmental values from profile
        solar_val = float(self.profile_df.loc[t, "solar_generation_kw"])
        grid_val = float(self.profile_df.loc[t, "grid_demand_kw"])
        price_val = float(self.profile_df.loc[t, "electricity_price_per_kwh"])

        # 3. Solar bin
        if solar_val < SOLAR_BINS[0]:
            solar_cat = 0
        elif solar_val < SOLAR_BINS[1]:
            solar_cat = 1
        else:
            solar_cat = 2

        # 4. Grid demand bin
        grid_cat = 0 if grid_val < GRID_BINS[0] else 1

        # 5. Price bin
        if price_val < PRICE_BINS[0]:
            price_cat = 0
        elif price_val < PRICE_BINS[1]:
            price_cat = 1
        else:
            price_cat = 2

        # Active EVs count
        currently_present = [e for e in self.evs if e.is_present(t)]
        occupancy_cat = 0 if len(currently_present) < OCCUPANCY_BINS[0] else 1

        # 2 & 6: EV-specific SOC and urgency
        if ev is not None:
            # SOC bin
            if ev.current_soc < SOC_BINS[0]:
                soc_cat = 0
            elif ev.current_soc < SOC_BINS[1]:
                soc_cat = 1
            else:
                soc_cat = 2

            # Urgency bin
            urgency = ev.calculate_urgency(t)
            if urgency < URGENCY_BINS[0]:
                urgency_cat = 0
            elif urgency < URGENCY_BINS[1]:
                urgency_cat = 1
            else:
                urgency_cat = 2
        else:
            soc_cat = 2       # Target reached / inactive
            urgency_cat = 0   # No urgency

        return (time_cat, soc_cat, solar_cat, grid_cat, price_cat, urgency_cat, occupancy_cat)

    @staticmethod
    def state_to_index(state: Tuple[int, int, int, int, int, int, int]) -> int:
        """Encodes discrete state tuple into a unique linear integer index.

        Dimensions: 4 x 3 x 3 x 2 x 3 x 3 x 2 = 1,296 states.
        """
        c0, c1, c2, c3, c4, c5, c6 = state
        idx = c0
        idx = idx * 3 + c1
        idx = idx * 3 + c2
        idx = idx * 2 + c3
        idx = idx * 3 + c4
        idx = idx * 3 + c5
        idx = idx * 2 + c6
        return idx

    @staticmethod
    def index_to_state(idx: int) -> Tuple[int, int, int, int, int, int, int]:
        """Decodes unique linear integer index back into 7-element state tuple."""
        c6 = idx % 2; idx //= 2
        c5 = idx % 3; idx //= 3
        c4 = idx % 3; idx //= 3
        c3 = idx % 2; idx //= 2
        c2 = idx % 3; idx //= 3
        c1 = idx % 3; idx //= 3
        c0 = idx % 4
        return (c0, c1, c2, c3, c4, c5, c6)

    # ==========================================
    # Priority Scheduling & Eligible EVs
    # ==========================================
    def get_eligible_evs(self) -> List[EV]:
        """Returns active connected EVs that have not reached target SOC,
        sorted in descending order of Urgency (Highest Urgency First).
        """
        eligible = [ev for ev in self.evs if ev.is_eligible_to_charge(self.current_step)]
        # Sort by urgency descending; tiebreaker: arrival step
        eligible.sort(key=lambda ev: (ev.calculate_urgency(self.current_step), -ev.arrival_step), reverse=True)
        return eligible

    def get_priority_ev(self) -> Optional[EV]:
        """Returns the single highest-priority eligible EV at current step, if any."""
        eligible = self.get_eligible_evs()
        return eligible[0] if eligible else None

    # ==========================================
    # Simulation Step Execution
    # ==========================================
    def step(self, actions_dict: Dict[int, int]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Executes one 15-minute simulation time step.

        Parameters:
        -----------
        actions_dict : Dict[int, int]
            Mapping from ev_id to action index (0: 0kW, 1: 3.5kW, 2: 7kW).

        Returns:
        --------
        step_info : Dict[str, Any]
        total_reward : float
        done : bool
        info : Dict[str, Any]
        """
        if self.current_step >= NUM_TIME_STEPS:
            return {}, 0.0, True, {"completed_evs": self.completed_evs}

        t = self.current_step
        solar_available_kw = float(self.profile_df.loc[t, "solar_generation_kw"])
        base_grid_kw = float(self.profile_df.loc[t, "grid_demand_kw"])
        price_per_kwh = float(self.profile_df.loc[t, "electricity_price_per_kwh"])

        eligible_evs = self.get_eligible_evs()
        remaining_station_power_kw = self.station_power_limit_kw

        total_ev_power_kw = 0.0
        ev_charge_records = {}

        # 1. Allocate power to eligible EVs according to priority order and station limits
        for ev in eligible_evs:
            requested_action = actions_dict.get(ev.ev_id, 0)
            requested_power_kw = ACTION_POWER_MAP.get(requested_action, 0.0)

            # Enforce individual port limit and remaining station capacity limit
            permissible_power_kw = min(requested_power_kw, ev.max_charging_power_kw, remaining_station_power_kw)
            
            # Apply charge to EV
            charge_res = ev.charge(permissible_power_kw, TIME_INTERVAL_HOURS)
            actual_power_kw = charge_res["power_drawn_kw"]

            remaining_station_power_kw = max(0.0, remaining_station_power_kw - actual_power_kw)
            total_ev_power_kw += actual_power_kw
            ev_charge_records[ev.ev_id] = charge_res

        # 2. Energy balance: solar power is consumed first; remaining deficit from grid
        solar_power_used_kw = min(total_ev_power_kw, solar_available_kw)
        grid_power_for_ev_kw = max(0.0, total_ev_power_kw - solar_power_used_kw)
        total_grid_power_kw = base_grid_kw + grid_power_for_ev_kw

        # 3. Energy and cost calculations
        grid_energy_for_ev_kwh = grid_power_for_ev_kw * TIME_INTERVAL_HOURS
        solar_energy_used_kwh = solar_power_used_kw * TIME_INTERVAL_HOURS
        step_electricity_cost = grid_energy_for_ev_kwh * price_per_kwh

        # 4. Check for departing EVs in this step and calculate unmet penalties
        step_unmet_energy_kwh = 0.0
        for ev in self.evs:
            # If EV departs at step t + 1 (or current step is departure_step - 1)
            if ev.departure_step == t + 1 and not ev.is_completed:
                session_summary = ev.finalize_session()
                self.completed_evs.append(session_summary)
                step_unmet_energy_kwh += session_summary["unmet_energy_kwh"]

        # 5. Compute scalar reward
        reward_dict = self.reward_fn.calculate_step_reward(
            grid_electricity_cost=step_electricity_cost,
            solar_energy_used_kwh=solar_energy_used_kwh,
            total_grid_power_kw=total_grid_power_kw,
            unmet_energy_kwh=step_unmet_energy_kwh,
            is_invalid_action=False,
        )
        total_step_reward = reward_dict["total_reward"]

        # 6. Record step history
        self.history["step"].append(t)
        self.history["time_hours"].append(round(t * 0.25, 2))
        self.history["solar_generation_kw"].append(solar_available_kw)
        self.history["base_grid_demand_kw"].append(base_grid_kw)
        self.history["total_grid_demand_kw"].append(round(total_grid_power_kw, 2))
        self.history["electricity_price"].append(price_per_kwh)
        self.history["total_ev_power_kw"].append(round(total_ev_power_kw, 2))
        self.history["solar_power_used_kw"].append(round(solar_power_used_kw, 2))
        self.history["grid_power_for_ev_kw"].append(round(grid_power_for_ev_kw, 2))
        self.history["step_electricity_cost"].append(round(step_electricity_cost, 4))
        self.history["step_reward"].append(total_step_reward)
        self.history["active_ev_count"].append(len(eligible_evs))

        # Advance step
        self.current_step += 1
        done = self.current_step >= NUM_TIME_STEPS

        # Finalize any remaining unfinalized EVs if done
        if done:
            for ev in self.evs:
                if not ev.is_completed:
                    session_summary = ev.finalize_session()
                    self.completed_evs.append(session_summary)

        step_info = {
            "step": t,
            "total_ev_power_kw": total_ev_power_kw,
            "solar_power_used_kw": solar_power_used_kw,
            "grid_power_for_ev_kw": grid_power_for_ev_kw,
            "grid_electricity_cost": step_electricity_cost,
            "reward_breakdown": reward_dict,
            "ev_charge_records": ev_charge_records,
        }

        return step_info, total_step_reward, done, {"completed_evs": self.completed_evs}

    def get_episode_metrics(self) -> Dict[str, float]:
        """Calculates final aggregated metrics for the completed 24-hour simulation episode."""
        dt = TIME_INTERVAL_HOURS

        total_grid_energy_kwh = sum(self.history["grid_power_for_ev_kw"]) * dt
        total_solar_energy_kwh = sum(self.history["solar_power_used_kw"]) * dt
        total_ev_energy_kwh = total_grid_energy_kwh + total_solar_energy_kwh
        total_cost = sum(self.history["step_electricity_cost"])
        peak_grid_demand_kw = max(self.history["total_grid_demand_kw"]) if self.history["total_grid_demand_kw"] else 0.0

        solar_contribution_pct = (
            (total_solar_energy_kwh / total_ev_energy_kwh * 100.0) if total_ev_energy_kwh > 0 else 0.0
        )

        # EV departure satisfaction metrics
        if self.completed_evs:
            departure_socs = [e["departure_soc"] for e in self.completed_evs]
            achieved_targets = [1.0 if e["achieved_target"] else 0.0 for e in self.completed_evs]
            unmet_energies = [e["unmet_energy_kwh"] for e in self.completed_evs]

            avg_departure_soc = float(np.mean(departure_socs))
            target_soc_achievement_pct = float(np.mean(achieved_targets) * 100.0)
            total_unmet_energy_kwh = float(np.sum(unmet_energies))
        else:
            avg_departure_soc = 0.0
            target_soc_achievement_pct = 0.0
            total_unmet_energy_kwh = 0.0

        total_reward = sum(self.history["step_reward"])

        return {
            "total_grid_electricity_cost": round(total_cost, 2),
            "total_grid_energy_kwh": round(total_grid_energy_kwh, 2),
            "total_solar_energy_kwh": round(total_solar_energy_kwh, 2),
            "solar_contribution_pct": round(solar_contribution_pct, 2),
            "peak_grid_demand_kw": round(peak_grid_demand_kw, 2),
            "avg_departure_soc": round(avg_departure_soc, 2),
            "target_soc_achievement_pct": round(target_soc_achievement_pct, 2),
            "total_unmet_energy_kwh": round(total_unmet_energy_kwh, 2),
            "total_ev_energy_kwh": round(total_ev_energy_kwh, 2),
            "total_episode_reward": round(total_reward, 2),
        }
