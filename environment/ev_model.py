"""Electric Vehicle (EV) dynamics model.
Tracks individual battery SOC, physical constraints, arrival/departure dwell times, and urgency.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
from config import (
    EV_BATTERY_CAPACITY_KWH,
    EV_MAX_CHARGING_POWER_KW,
    EV_CHARGING_EFFICIENCY,
    EV_TARGET_SOC,
    EV_MAX_SOC,
    TIME_INTERVAL_HOURS,
)


@dataclass
class EV:
    """Represents a single Electric Vehicle arriving at the charging station."""
    ev_id: int
    arrival_step: int
    departure_step: int
    initial_soc: float
    target_soc: float = EV_TARGET_SOC
    battery_capacity_kwh: float = EV_BATTERY_CAPACITY_KWH
    max_charging_power_kw: float = EV_MAX_CHARGING_POWER_KW
    charging_efficiency: float = EV_CHARGING_EFFICIENCY

    current_soc: float = field(init=False)
    total_energy_charged_kwh: float = field(default=0.0, init=False)
    total_energy_drawn_kwh: float = field(default=0.0, init=False)
    is_completed: bool = field(default=False, init=False)
    departure_soc: float = field(default=0.0, init=False)

    def __post_init__(self):
        # Validate logic
        if self.departure_step <= self.arrival_step:
            raise ValueError(f"Departure step ({self.departure_step}) must be strictly greater than arrival ({self.arrival_step})")
        self.initial_soc = float(max(0.0, min(EV_MAX_SOC, self.initial_soc)))
        self.current_soc = self.initial_soc
        self.departure_soc = self.initial_soc

    def is_present(self, current_step: int) -> bool:
        """Returns True if the EV is currently connected at the station."""
        return self.arrival_step <= current_step < self.departure_step

    def is_eligible_to_charge(self, current_step: int) -> bool:
        """Returns True if EV is present and has not yet achieved target SOC."""
        return self.is_present(current_step) and (self.current_soc < self.target_soc)

    def is_departing(self, current_step: int) -> bool:
        """Returns True if this time step is the exact departure step of the EV."""
        return current_step == self.departure_step

    def calculate_urgency(self, current_step: int) -> float:
        """Computes departure urgency ratio.

        Defined as:
            (Energy needed to reach target SOC) / (Max deliverable energy before departure)

        Values:
        - 0.0: Target already achieved
        - < 0.40: Low urgency (abundant dwell time available)
        - 0.40 - 0.75: Moderate urgency
        - >= 0.75: High urgency (EV must charge promptly to meet departure requirement)
        """
        if not self.is_present(current_step) or self.current_soc >= self.target_soc:
            return 0.0

        remaining_steps = max(1, self.departure_step - current_step)
        soc_deficit = max(0.0, self.target_soc - self.current_soc)
        energy_needed_kwh = (soc_deficit / 100.0) * self.battery_capacity_kwh

        # Maximum energy battery can absorb before departure under max power
        max_possible_charge_kwh = (
            self.max_charging_power_kw * TIME_INTERVAL_HOURS * self.charging_efficiency * remaining_steps
        )
        if max_possible_charge_kwh <= 1e-6:
            return 2.0  # Critically overdue

        return min(2.5, energy_needed_kwh / max_possible_charge_kwh)

    def charge(self, power_kw: float, time_interval_h: float = TIME_INTERVAL_HOURS) -> Dict[str, float]:
        """Applies charging power to battery using physical conservation laws.

        Formula:
            Delta_SOC = (Power * dt * Efficiency / Capacity) * 100%

        Ensures:
        - Power is non-negative and capped at EV maximum power limit.
        - SOC never exceeds 100.0%.
        - Actual power delivered matches available battery headroom.
        """
        if power_kw <= 0.0:
            return {"power_drawn_kw": 0.0, "power_delivered_kw": 0.0, "energy_drawn_kwh": 0.0, "energy_stored_kwh": 0.0, "delta_soc": 0.0}

        # Physical limit check
        requested_power = min(self.max_charging_power_kw, max(0.0, power_kw))

        # Check battery capacity headroom to 100%
        remaining_capacity_percent = max(0.0, EV_MAX_SOC - self.current_soc)
        max_deliverable_energy_kwh = (remaining_capacity_percent / 100.0) * self.battery_capacity_kwh

        # Energy drawn from station vs stored in battery
        energy_attempted_drawn_kwh = requested_power * time_interval_h
        energy_attempted_stored_kwh = energy_attempted_drawn_kwh * self.charging_efficiency

        # Clip if battery would exceed 100%
        actual_energy_stored_kwh = min(max_deliverable_energy_kwh, energy_attempted_stored_kwh)
        actual_energy_drawn_kwh = actual_energy_stored_kwh / self.charging_efficiency if self.charging_efficiency > 0 else 0.0
        actual_power_drawn_kw = actual_energy_drawn_kwh / time_interval_h

        # Update SOC
        delta_soc = (actual_energy_stored_kwh / self.battery_capacity_kwh) * 100.0
        self.current_soc = min(EV_MAX_SOC, self.current_soc + delta_soc)

        self.total_energy_drawn_kwh += actual_energy_drawn_kwh
        self.total_energy_charged_kwh += actual_energy_stored_kwh
        self.departure_soc = self.current_soc

        return {
            "power_drawn_kw": actual_power_drawn_kw,
            "power_delivered_kw": actual_power_drawn_kw * self.charging_efficiency,
            "energy_drawn_kwh": actual_energy_drawn_kwh,
            "energy_stored_kwh": actual_energy_stored_kwh,
            "delta_soc": delta_soc,
        }

    def finalize_session(self) -> Dict[str, Any]:
        """Finalizes EV metrics upon departure."""
        self.departure_soc = self.current_soc
        self.is_completed = True
        soc_deficit = max(0.0, self.target_soc - self.departure_soc)
        unmet_energy_kwh = (soc_deficit / 100.0) * self.battery_capacity_kwh

        return {
            "ev_id": self.ev_id,
            "initial_soc": self.initial_soc,
            "target_soc": self.target_soc,
            "departure_soc": self.departure_soc,
            "achieved_target": self.departure_soc >= (self.target_soc - 0.1),
            "unmet_energy_kwh": unmet_energy_kwh,
            "total_drawn_kwh": self.total_energy_drawn_kwh,
            "total_charged_kwh": self.total_energy_charged_kwh,
        }

