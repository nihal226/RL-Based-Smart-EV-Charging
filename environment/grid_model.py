"""Grid model for baseline electricity demand and time-varying electricity tariffs.
Reflects diurnal commercial/residential loading and dynamic Time-Of-Use (TOU) tariffs.
"""

import numpy as np
from typing import Optional, Tuple
from config import (
    NUM_TIME_STEPS,
    PRICE_OFF_PEAK,
    PRICE_MID_PEAK,
    PRICE_ON_PEAK,
    PRICE_EVENING_MID,
)


class GridModel:
    """Generates realistic synthetic 24-hour background grid load and TOU price profiles."""

    def __init__(self, base_load_kw: float = 16.0, peak_load_kw: float = 46.0):
        self.base_load_kw = base_load_kw
        self.peak_load_kw = peak_load_kw

    def generate_demand_profile(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Generates a 96-step (15-min interval) baseline grid demand profile (kW).

        Pattern:
        - Overnight (00:00 - 05:30): Low base demand (~15 - 18 kW)
        - Morning peak ramp (06:00 - 10:00): Rising commercial/commuter demand (~28 - 34 kW)
        - Midday plateau (11:00 - 16:00): Moderate daytime activity (~24 - 28 kW)
        - Evening peak (17:00 - 21:00): Peak residential arrival & cooking/appliances (~42 - 50 kW)
        - Late night wind-down (21:30 - 24:00): Returning to base load (~20 - 16 kW)
        """
        if rng is None:
            rng = np.random.default_rng()

        hours = np.arange(NUM_TIME_STEPS) * 0.25

        # Multi-modal baseline curve constructed from Gaussians representing morning & evening peaks
        morning_peak = 14.0 * np.exp(-((hours - 8.5) ** 2) / (2 * 1.8 ** 2))
        evening_peak = 28.0 * np.exp(-((hours - 19.0) ** 2) / (2 * 2.2 ** 2))
        daytime_offset = 8.0 * (1.0 / (1.0 + np.exp(-(hours - 7.0))) - 1.0 / (1.0 + np.exp(-(hours - 21.5))))

        deterministic_demand = self.base_load_kw + morning_peak + evening_peak + daytime_offset

        # Stochastic noise with temporal persistence (moving average)
        noise = rng.normal(loc=0.0, scale=1.5, size=NUM_TIME_STEPS)
        smoothed_noise = np.convolve(noise, np.ones(3) / 3.0, mode='same')
        demand_profile = deterministic_demand + smoothed_noise

        # Physical sanity check: non-negative and realistic minimum load
        demand_profile = np.maximum(5.0, demand_profile)
        return np.round(demand_profile, 2)

    def generate_price_profile(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Generates a 96-step Time-of-Use (TOU) electricity pricing profile ($/kWh).

        Tariff bands:
        - 00:00 - 06:00 (steps 0 - 23): Off-peak ($0.08/kWh)
        - 06:00 - 16:00 (steps 24 - 63): Mid-peak ($0.15/kWh)
        - 16:00 - 21:00 (steps 64 - 83): On-peak ($0.32/kWh)
        - 21:00 - 23:00 (steps 84 - 91): Evening mid-peak ($0.18/kWh)
        - 23:00 - 24:00 (steps 92 - 95): Off-peak ($0.08/kWh)

        A small stochastic jitter (+- 5%) across episodes mimics real-time wholesale volatility.
        """
        if rng is None:
            rng = np.random.default_rng()

        prices = np.zeros(NUM_TIME_STEPS, dtype=float)

        # Off-peak early morning
        prices[0:24] = PRICE_OFF_PEAK
        # Mid-peak morning & daytime
        prices[24:64] = PRICE_MID_PEAK
        # On-peak late afternoon/evening
        prices[64:84] = PRICE_ON_PEAK
        # Evening mid-peak
        prices[84:92] = PRICE_EVENING_MID
        # Off-peak late night
        prices[92:96] = PRICE_OFF_PEAK

        # Subtle market fluctuation (+- 3-5%) with temporal smoothing
        jitter = rng.normal(loc=0.0, scale=0.005, size=NUM_TIME_STEPS)
        smoothed_jitter = np.convolve(jitter, np.ones(3) / 3.0, mode='same')
        dynamic_prices = prices + smoothed_jitter

        # Sanity check: keep strictly positive and bounded
        dynamic_prices = np.clip(dynamic_prices, 0.05, 0.45)
        return np.round(dynamic_prices, 4)

    def generate_grid_scenario(self, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience method returning both demand and price profiles."""
        return self.generate_demand_profile(rng), self.generate_price_profile(rng)

