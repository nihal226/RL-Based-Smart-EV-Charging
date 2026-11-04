"""Baseline 2: Solar-First (Renewable-Prioritized) Charging Strategy.
Maximizes direct on-site solar PV self-consumption.
Restricts grid charging to situations where EV departure deadlines are imminent (high urgency).
"""

from typing import Dict
from environment.ev_environment import EVChargingEnv
from config import URGENCY_BINS, PRICE_BINS


class SolarFirstChargingPolicy:
    """Heuristic rule-based policy prioritizing renewable solar energy."""

    def __init__(self, name: str = "Solar-First Charging"):
        self.name = name

    def get_actions(self, env: EVChargingEnv) -> Dict[int, int]:
        """Determines charging action for all active eligible EVs based on solar availability and deadline urgency.

        Logic:
        1. If abundant solar PV is generating, charge EVs aggressively (Action 2 or 1).
        2. If solar PV is scarce or zero:
           - If EV has High Urgency (urgency >= 0.75), charge from grid to satisfy target SOC before departure.
           - If EV has Moderate Urgency and electricity price is low/off-peak, charge slowly (Action 1 = 3.5 kW).
           - Otherwise, defer charging and wait for daytime solar generation (Action 0 = 0 kW).
        """
        t = env.current_step
        solar_kw = float(env.profile_df.loc[t, "solar_generation_kw"])
        price_kwh = float(env.profile_df.loc[t, "electricity_price_per_kwh"])

        eligible_evs = env.get_eligible_evs()
        actions = {}

        if not eligible_evs:
            return actions

        # Remaining solar power budget that can be absorbed
        solar_budget_kw = solar_kw

        for ev in eligible_evs:
            urgency = ev.calculate_urgency(t)

            # Scenario A: Generous solar available
            if solar_budget_kw >= 7.0:
                actions[ev.ev_id] = 2  # 7 kW
                solar_budget_kw -= 7.0
            elif solar_budget_kw >= 3.5:
                actions[ev.ev_id] = 1  # 3.5 kW
                solar_budget_kw -= 3.5
            else:
                # Scenario B: Solar is depleted or absent
                # Must we charge from grid to meet departure deadline?
                if urgency >= URGENCY_BINS[1]:  # High urgency (>= 0.75)
                    actions[ev.ev_id] = 2  # Fast charge to avert penalty
                elif urgency >= URGENCY_BINS[0] and price_kwh < PRICE_BINS[1]:
                    # Moderate urgency and reasonable price: slow charge
                    actions[ev.ev_id] = 1  # 3.5 kW
                else:
                    # Low urgency or expensive grid peak: wait for solar
                    actions[ev.ev_id] = 0  # 0 kW

        return actions

