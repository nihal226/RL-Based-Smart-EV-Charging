"""Baselines package."""
from baselines.immediate_charging import ImmediateChargingPolicy
from baselines.solar_first import SolarFirstChargingPolicy

__all__ = ["ImmediateChargingPolicy", "SolarFirstChargingPolicy"]
