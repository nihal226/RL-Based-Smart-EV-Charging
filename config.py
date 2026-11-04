"""Configuration parameters for the RL-Based Smart EV Charging System.
All time intervals, physical units, hyper-parameters, and file paths are centralized here.
"""

from pathlib import Path

# ==========================================
# Paths & Directories
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "training"
TEST_DATA_DIR = DATA_DIR / "testing"
PROFILES_DATA_DIR = DATA_DIR / "generated_profiles"

RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"

# Ensure essential output directories exist
for path in [TRAIN_DATA_DIR, TEST_DATA_DIR, PROFILES_DATA_DIR, PLOTS_DIR, METRICS_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# ==========================================
# Simulation Parameters
# ==========================================
DURATION_HOURS = 24
TIME_STEP_MINUTES = 15
NUM_TIME_STEPS = 96  # 24 * 60 / 15
TIME_INTERVAL_HOURS = TIME_STEP_MINUTES / 60.0  # 0.25 hours
RANDOM_SEED = 42

# ==========================================
# EV Technical Characteristics
# ==========================================
EV_BATTERY_CAPACITY_KWH = 40.0
EV_MAX_CHARGING_POWER_KW = 7.0
EV_CHARGING_EFFICIENCY = 0.90  # 90%
EV_INITIAL_SOC_MIN = 15.0  # %
EV_INITIAL_SOC_MAX = 70.0  # %
EV_TARGET_SOC = 80.0       # % Target departure SOC
EV_MAX_SOC = 100.0         # % Physical hard cap

# Typical EV arrival configuration
MIN_EVS_PER_DAY = 4
MAX_EVS_PER_DAY = 6
MIN_DWELL_HOURS = 2.0
MAX_DWELL_HOURS = 8.0

# ==========================================
# Charging Station Characteristics
# ==========================================
NUM_CHARGING_PORTS = 3
TOTAL_STATION_POWER_LIMIT_KW = 21.0  # 3 ports * 7 kW

# Action Space: 0 = 0 kW, 1 = 3.5 kW (slow), 2 = 7.0 kW (fast)
ACTIONS = [0, 1, 2]
ACTION_POWER_MAP = {
    0: 0.0,
    1: 3.5,
    2: 7.0,
}

# ==========================================
# Grid & Solar Profile Constants
# ==========================================
SOLAR_RATED_CAPACITY_KW = 18.0   # Station solar PV capacity
PEAK_GRID_THRESHOLD_KW = 35.0    # Grid demand threshold above which peak penalties apply

# TOU Price Levels ($/kWh)
PRICE_OFF_PEAK = 0.08
PRICE_MID_PEAK = 0.15
PRICE_ON_PEAK = 0.32
PRICE_EVENING_MID = 0.18

# ==========================================
# Reward Function Calibration
# ==========================================
# Reward = - alpha_cost * Cost - beta_peak * Peak_Penalty + gamma_solar * Solar_Used - delta_unmet * Unmet_Penalty
ALPHA_COST = 1.0        # Weight for grid electricity cost ($)
BETA_PEAK = 0.3         # Weight for peak grid demand penalty
GAMMA_SOLAR = 1.2       # Weight for solar energy utilized (kWh)
DELTA_UNMET = 4.0       # Weight for unmet departure energy penalty (kWh deficit)
INVALID_ACTION_PENALTY = 2.0  # Negative reward if agent attempts an impermissible action

# ==========================================
# Tabular Q-Learning Hyperparameters
# ==========================================
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.995
TRAINING_EPISODES = 2000
EVAL_EPISODES = 20

# Parameter Analysis Experiments
LEARNING_RATE_EXPERIMENTS = [0.01, 0.1, 0.5]
SOLAR_REDUCTION_FACTOR = 0.70  # 30% reduction in solar availability

# ==========================================
# State Space Discretization Categories
# ==========================================
# 1. Time category (4 bins)
#    0: Night (00:00 - 06:00, steps 0-23)
#    1: Morning (06:00 - 11:00, steps 24-43)
#    2: Midday (11:00 - 16:00, steps 44-63)
#    3: Evening (16:00 - 24:00, steps 64-95)
TIME_BINS = [24, 44, 64]

# 2. EV SOC category (3 bins)
#    0: Low (< 50%)
#    1: Medium (50% <= SOC < 75%)
#    2: Target/High (SOC >= 75%)
SOC_BINS = [50.0, 75.0]

# 3. Solar availability category (3 bins)
#    0: Low (< 2.0 kW)
#    1: Medium (2.0 <= Solar < 8.0 kW)
#    2: High (Solar >= 8.0 kW)
SOLAR_BINS = [2.0, 8.0]

# 4. Grid demand category (2 bins)
#    0: Normal (< 35.0 kW)
#    1: Peak (>= 35.0 kW)
GRID_BINS = [35.0]

# 5. Electricity price category (3 bins)
#    0: Low (< $0.12/kWh)
#    1: Medium ($0.12 <= Price < $0.22/kWh)
#    2: High (Price >= $0.22/kWh)
PRICE_BINS = [0.12, 0.22]

# 6. Departure urgency category (3 bins)
#    Urgency metric: Ratio of remaining required energy to maximum deliverable energy before departure
#    0: Low (< 0.40) - ample time
#    1: Moderate (0.40 <= Urgency < 0.75)
#    2: High (Urgency >= 0.75) - urgent charging required
URGENCY_BINS = [0.40, 0.75]

# 7. Station occupancy category (2 bins)
#    0: Low (0 or 1 active EV plugged in)
#    1: High (2 or 3 active EVs plugged in)
OCCUPANCY_BINS = [2]

