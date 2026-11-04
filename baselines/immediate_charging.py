"""Baseline 1: Immediate (Greedy / Uncontrolled) Charging Strategy.
Charges every eligible connected EV at maximum allowable power (7 kW)
immediately upon arrival until target SOC is reached, subject only to station capacity.
Does not optimize for Time-of-Use pricing or on-site solar availability.
"""

from typing import Dict, List
from environment.ev_environment import EVChargingEnv
from environment.ev_model import EV


class ImmediateChargingPolicy:
    """Conventional uncontrolled immediate charging heuristic."""

    def __init__(self, name: str = "Immediate Charging"):
        self.name = name

    def get_actions(self, env: EVChargingEnv) -> Dict[int, int]:
        """Determines charging action for all active eligible EVs.

        Logic:
        - For every eligible connected EV, request Action 2 (7.0 kW).
        - Environment priority scheduler handles port and total station capacity (21 kW).
        """
        eligible_evs = env.get_eligible_evs()
        actions = {}
        for ev in eligible_evs:
            # Always request maximum fast charging (Action 2 = 7.0 kW)
            actions[ev.ev_id] = 2

        return actions

