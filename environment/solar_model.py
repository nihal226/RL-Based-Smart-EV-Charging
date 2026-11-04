"""Solar generation model for simulating realistic 24-hour PV production.
Implements diurnal solar variation with stochastic weather/cloud attenuation.
"""

import numpy as np
from typing import Optional
from config import NUM_TIME_STEPS, SOLAR_RATED_CAPACITY_KW


class SolarModel:
    """Simulates realistic 24-hour solar photovoltaic power profiles."""

    def __init__(self, rated_capacity_kw: float = SOLAR_RATED_CAPACITY_KW):
        self.rated_capacity_kw = rated_capacity_kw

    def generate_profile(self, rng: Optional[np.random.Generator] = None, cloudiness_factor: Optional[float] = None) -> np.ndarray:
        """Generates a 96-step (15-min interval) solar power profile (kW).

        Parameters:
        -----------
        rng : np.random.Generator, optional
            Numpy random generator for reproducible stochasticity.
        cloudiness_factor : float, optional
            Weather factor [0.0 = completely overcast, 1.0 = clear sky].
            If None, generated stochastically.

        Returns:
        --------
        np.ndarray: 96-element array of solar power generation in kW.
        """
        if rng is None:
            rng = np.random.default_rng()

        time_steps = np.arange(NUM_TIME_STEPS)
        hours = time_steps * 0.25  # 0.0 to 23.75

        # Sunrise around 06:00 (step 24), Sunset around 18:30 (step 74)
        sunrise_hour = 6.0
        sunset_hour = 18.5
        solar_noon_hour = 12.25

        # Clear-sky bell-shaped curve using half-sine between sunrise and sunset
        clear_sky_profile = np.zeros(NUM_TIME_STEPS, dtype=float)
        daylight_mask = (hours >= sunrise_hour) & (hours <= sunset_hour)

        daylight_hours = hours[daylight_mask]
        # Angle from 0 at sunrise to pi at sunset
        phase = np.pi * (daylight_hours - sunrise_hour) / (sunset_hour - sunrise_hour)
        # Power proportional to sin(phase) with slight midday peak convexity
        clear_sky_profile[daylight_mask] = self.rated_capacity_kw * np.sin(phase) ** 1.1

        # Determine overall day condition if not explicitly provided
        if cloudiness_factor is None:
            # Weather classification: 60% sunny/mostly sunny, 30% partly cloudy, 10% overcast
            weather_draw = rng.uniform(0.0, 1.0)
            if weather_draw > 0.40:
                day_attenuation = rng.uniform(0.85, 1.0)  # Clear/mostly sunny
            elif weather_draw > 0.10:
                day_attenuation = rng.uniform(0.50, 0.85)  # Partly cloudy
            else:
                day_attenuation = rng.uniform(0.20, 0.50)  # Overcast
        else:
            day_attenuation = np.clip(cloudiness_factor, 0.0, 1.0)

        # Stochastic passing cloud fluctuations (temporally smoothed noise)
        noise = rng.normal(loc=0.0, scale=0.08, size=NUM_TIME_STEPS)
        # 3-point moving average smoothing for realistic cloud persistence
        smoothed_noise = np.convolve(noise, np.ones(3) / 3.0, mode='same')
        cloud_multipliers = np.clip(day_attenuation + smoothed_noise, 0.05, 1.05)

        solar_profile = clear_sky_profile * cloud_multipliers

        # Strictly non-negative and zero outside daylight hours
        solar_profile[~daylight_mask] = 0.0
        solar_profile = np.maximum(0.0, solar_profile)

        return np.round(solar_profile, 3)

