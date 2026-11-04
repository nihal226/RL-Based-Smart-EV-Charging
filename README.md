# Reinforcement Learning-Based Smart EV Charging System
## Adaptive Charging Scheduling Under Dynamic Grid Demand and Solar PV Availability

---

## 1. Project Overview

This repository presents an academic-grade, reproducible simulation and Tabular Reinforcement Learning (RL) framework for smart Electric Vehicle (EV) charging scheduling. The system models a 24-hour commercial/workplace charging station equipped with on-site Solar Photovoltaic (PV) generation, dynamic Time-of-Use (TOU) electricity pricing, stochastic background grid demand, and physical battery electrochemical constraints.

The objective is to train an autonomous Reinforcement Learning agent using **Tabular Q-Learning** that adaptively schedules charging rates for arriving EVs to minimize operating electricity costs, maximize clean solar self-consumption, mitigate peak grid loading, and ensure every EV departs with its required battery State of Charge (SOC $\ge 80\%$).

---

## 2. Problem Statement

Uncontrolled EV charging ("Immediate / Dumb Charging") introduces severe strains on electrical distribution networks, characterized by:
- Coinciding with evening residential peak demand hours (17:00 – 21:00), risking substation transformer overloading and voltage drops.
- High electricity purchasing costs due to peak Time-of-Use tariffs.
- Low utilization of local renewable solar energy generated during daytime hours (10:00 – 15:00).
- Suboptimal utilization of station power limits.

Conversely, heuristic solar-prioritized charging can result in EVs departing with deficient battery charge if solar generation fluctuates due to passing cloud cover or seasonal variations.

An optimal charging scheduler must dynamically balance multiple competing objectives:
1. Economic efficiency (minimizing grid electricity procurement costs).
2. Grid resilience (peak load shaving and avoiding transformer capacity penalties).
3. Environmental sustainability (maximizing on-site solar self-consumption).
4. User quality of service (QoS) and satisfaction (meeting required departure SOC).

---

## 3. Objectives

- **Adaptive Scheduling**: Learn an optimal discrete charging policy without prior mathematical models of future solar irradiance or grid prices.
- **Physical Realism**: Model battery capacity ($C_{\text{battery}} = 40\text{ kWh}$), charging efficiency ($\eta = 90\%$), charging power limits ($P_{\text{max}} = 7\text{ kW}$), and total station limits ($P_{\text{station}} = 21\text{ kW}$).
- **Benchmarking**: Compare the proposed Q-learning agent against two conventional baselines (Immediate Charging and Solar-First Charging) under identical test scenarios.
- **Sensitivity & Robustness**: Assess performance under hyperparameter variations ($\alpha \in \{0.01, 0.1, 0.5\}$) and environmental stress (30% solar reduction).
- **Reproducibility**: Ensure seed-based reproducibility and modular architecture with zero external dataset dependencies.

---

## 4. Literature Review Guidance

In energy systems and smart grid research, EV charging scheduling is typically addressed through:
1. **Mathematical Programming (MILP / MPC)**: Solves deterministic or rolling-horizon optimization. Requires accurate forecasts of solar irradiance, spot prices, and EV dwell times. Sensitive to forecast errors.
2. **Rule-Based / Heuristic Scheduling**: Simple to deploy (e.g., First-Come-First-Served, Solar-First, or Threshold-based). Fails to adapt to dynamic multi-objective trade-offs.
3. **Reinforcement Learning (RL)**: Formulates charging as a Markov Decision Process (MDP). Tabular Q-learning is model-free, computationally lightweight, transparent, and provably convergent under finite state-action spaces without the instability or black-box opacity of Deep Q-Networks (DQN).

Key seminal references to consult for academic writing:
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- Garcia-Villalobos et al. (2014). *Plug-in electric vehicles in electric distribution networks: A review of smart charging approaches*. Renewable and Sustainable Energy Reviews.
- Mwasilu et al. (2014). *Electric vehicles and smart grid interaction: A review on vehicle to grid and renewable energy sources integration*. Renewable and Sustainable Energy Reviews.

---

## 5. System Architecture

```
                                  [ Solar PV Generation ]
                                             │
                                             ▼
[ Grid Tariff & Demand ] ───► [ Smart Charging Station ] ◄─── [ Arriving EVs ]
                                  (3 Ports, 21 kW Limit)       (SOC: 15% - 70%)
                                             │
                                             ▼
                                  [ Tabular Q-Learning ]
                                   (State Discretization)
                                             │
                                             ▼
                             [ Optimal Charging Dispatch ]
                             (0 kW, 3.5 kW, 7.0 kW per EV)
```

The system operates on a 24-hour simulation horizon with 15-minute resolution ($N = 96$ time steps per day, $\Delta t = 0.25\text{ h}$).

---

## 6. Markov Decision Process (MDP) Formulation

The smart EV charging scheduling problem is formulated as a discrete-time Markov Decision Process:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

Where:
- $\mathcal{S}$ is the discrete state space.
- $\mathcal{A}$ is the discrete action space.
- $\mathcal{P}(s' \mid s, a)$ represents the state transition probability distribution governed by the environment dynamics.
- $\mathcal{R}(s, a, s')$ is the multi-objective scalar reward function.
- $\gamma \in [0, 1)$ is the temporal discount factor ($\gamma = 0.95$).

---

## 7. State Space Formulation

To ensure exact tabular convergence and avoid the curse of dimensionality, the continuous state variables are discretized into **1,296 unique states** ($4 \times 3 \times 3 \times 2 \times 3 \times 3 \times 2$):

| Feature | Description | Discretization Thresholds / Categories | Bins |
| :--- | :--- | :--- | :---: |
| **1. Time of Day ($t$)** | Current step category | 0: Night (00:00–06:00), 1: Morning (06:00–11:00),<br>2: Midday (11:00–16:00), 3: Evening (16:00–24:00) | 4 |
| **2. EV Battery SOC** | State of charge of priority EV | 0: Low ($< 50\%$), 1: Medium ($50\% \le \text{SOC} < 75\%$),<br>2: Target Reached / High ($\ge 75\%$) | 3 |
| **3. Solar Availability ($P_{\text{solar}}$)** | Real-time solar power generation | 0: Low / None ($< 2.0\text{ kW}$), 1: Moderate ($2.0–8.0\text{ kW}$),<br>2: High ($\ge 8.0\text{ kW}$) | 3 |
| **4. Grid Demand ($P_{\text{grid}}$)** | Background grid loading | 0: Normal ($< 35.0\text{ kW}$), 1: Peak Stress ($\ge 35.0\text{ kW}$) | 2 |
| **5. Electricity Price ($c$)** | Real-time TOU tariff rate | 0: Off-Peak ($< \$0.12/\text{kWh}$), 1: Mid-Peak ($\$0.12–\$0.22/\text{kWh}$),<br>2: On-Peak ($\ge \$0.22/\text{kWh}$) | 3 |
| **6. Departure Urgency ($U$)** | Ratio of needed energy to maximum deliverable energy | 0: Low ($< 0.40$), 1: Moderate ($0.40 \le U < 0.75$),<br>2: High ($U \ge 0.75$) | 3 |
| **7. Station Occupancy ($N_{\text{ev}}$)** | Number of actively plugged-in EVs | 0: Low ($\le 1\text{ EV}$), 1: High ($\ge 2\text{ EVs}$) | 2 |

### Departure Urgency Index
$$U(t) = \frac{\max\left(0, \frac{\text{SOC}_{\text{target}} - \text{SOC}(t)}{100} \cdot C_{\text{battery}}\right)}{P_{\text{max}} \cdot \Delta t \cdot \eta \cdot (t_{\text{dep}} - t) + \epsilon}$$
- $U < 0.40$: Ample dwell time remaining; charging can be safely deferred to solar or off-peak hours.
- $U \ge 0.75$: High urgency; the EV must charge immediately to avoid departure deficits.

---

## 8. Action Space & Priority Scheduling

For each eligible EV, the action space consists of 3 discrete charging levels:
$$\mathcal{A} = \{0, 1, 2\}$$
- **Action 0**: $0.0\text{ kW}$ (No charging / Idle).
- **Action 1**: $3.5\text{ kW}$ (Slow / Level-1 AC charging).
- **Action 2**: $7.0\text{ kW}$ (Fast / Level-2 AC charging at maximum port rating).

### Priority Scheduling Mechanism (Highest Urgency First)
When multiple EVs are connected simultaneously:
1. Eligible EVs ($\text{SOC} < 80\%$) are sorted in descending order of Urgency Index $U(t)$ (tie-breaker: earlier arrival time).
2. The policy assigns power allocations sequentially.
3. The environment strictly enforces:
   - Individual EV maximum limit: $P_i \le 7.0\text{ kW}$.
   - Port limit: at most 3 EVs plugged in concurrently.
   - Aggregate station power capacity: $\sum_{i=1}^{N_t} P_i(t) \le 21.0\text{ kW}$.

---

## 9. Mathematically Justified Reward Function

The reward function guides the agent to simultaneously minimize procurement expenses, shave peak grid spikes, consume renewable solar power, and satisfy driver SOC requirements:

$$\mathcal{R}_t = - \alpha_{\text{cost}} \cdot C_{\text{grid}}(t) - \beta_{\text{peak}} \cdot \Omega_{\text{peak}}(t) + \gamma_{\text{solar}} \cdot E_{\text{solar}}(t) - \delta_{\text{unmet}} \cdot E_{\text{unmet}}(t)$$

### Breakdown of Reward Components
1. **Grid Electricity Cost Penalty**:
   $$C_{\text{grid}}(t) = P_{\text{grid\_ev}}(t) \cdot \Delta t \cdot c_{\text{grid}}(t) \quad [\$]$$
   Penalizes monetary expenditure for grid electricity drawn by EVs. Weight: $\alpha_{\text{cost}} = 1.0$.

2. **Peak Grid Demand Penalty**:
   $$\Omega_{\text{peak}}(t) = \beta_{\text{peak}} \cdot \frac{\max\left(0, P_{\text{total\_grid}}(t) - P_{\text{threshold}}\right)}{10.0}$$
   Penalizes total grid loading exceeding the substation capacity threshold ($35\text{ kW}$). Weight: $\beta_{\text{peak}} = 0.3$.

3. **Solar Energy Utilization Reward**:
   $$E_{\text{solar}}(t) = P_{\text{solar\_used}}(t) \cdot \Delta t \quad [\text{kWh}]$$
   Positive incentive for directly utilizing clean on-site solar power. Weight: $\gamma_{\text{solar}} = 1.2$.

4. **Unmet Departure SOC Penalty**:
   $$E_{\text{unmet}} = \sum_{j \in \text{departing}} \max\left(0, \frac{\text{SOC}_{\text{target}} - \text{SOC}_{j,\text{dep}}}{100}\right) \cdot C_{\text{battery}} \quad [\text{kWh}]$$
   Severe penalty assessed upon EV departure. Weight: $\delta_{\text{unmet}} = 4.0$. Because $1\text{ kWh}$ of unmet energy costs $\$4.00$ in penalty while grid electricity costs at most $\$0.32/\text{kWh}$, the agent is strictly prevented from "reward hacking" (i.e. leaving vehicles uncharged to save power costs).

---

## 10. Transition Dynamics & Battery Electrochemical Model

For each active vehicle $i$ at time step $t$:
$$\Delta \text{SOC}_i(t) = \left( \frac{P_{i,\text{actual}}(t) \cdot \Delta t \cdot \eta}{C_{\text{battery}}} \right) \times 100\%$$
$$\text{SOC}_i(t+1) = \min\left(100.0\%, \text{SOC}_i(t) + \Delta \text{SOC}_i(t)\right)$$

Where:
- $\Delta t = 0.25\text{ hours}$ (15 minutes).
- $\eta = 0.90$ (90% round-trip charging efficiency).
- $C_{\text{battery}} = 40.0\text{ kWh}$.
- Hard upper bound: $\text{SOC} \le 100\%$.
- Non-negativity: $\text{SOC} \ge 0\%$.
- Automatic cutoff: Charging ceases once target SOC ($80\%$) is reached.

### Energy Balance Equations
- $P_{\text{ev\_total}}(t) = \sum_{i} P_{i,\text{actual}}(t)$
- $P_{\text{solar\_used}}(t) = \min(P_{\text{ev\_total}}(t), P_{\text{solar}}(t))$
- $P_{\text{grid\_ev}}(t) = \max(0, P_{\text{ev\_total}}(t) - P_{\text{solar\_used}}(t))$
- $P_{\text{total\_grid}}(t) = P_{\text{base\_grid}}(t) + P_{\text{grid\_ev}}(t)$

---

## 11. Tabular Q-Learning Algorithm

The state-action value function $Q: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ is iteratively updated via the Bellman Optimality Equation:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a' \in \mathcal{A}} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

### Exploration via Epsilon Decay
$$\epsilon_{k+1} = \max\left(\epsilon_{\text{min}}, \epsilon_k \cdot \lambda\right)$$
- Initial $\epsilon_0 = 1.0$ (pure exploration).
- Decay rate $\lambda = 0.995$ per episode.
- Minimum $\epsilon_{\text{min}} = 0.01$ (1% exploration floor).
- Evaluation mode: $\epsilon = 0.0$ (strictly greedy exploitation, Q-table frozen).

---

## 12. Synthetic Data Generation Methodology

100 independent 24-hour simulation profiles (80 training, 20 testing) are generated programmatically:

1. **Solar Photovoltaic Profile**:
   - Diurnal half-sine profile between sunrise (06:00) and sunset (18:30) peaking at noon ($18\text{ kW}$ rated capacity).
   - Stochastic passing cloud attenuation using smoothed temporally correlated Gaussian noise.
   - Zero production at night.

2. **Baseline Grid Demand Profile**:
   - Dual-peak Gaussian distribution modeling morning ramp ($07:30–09:30$) and evening residential peak ($17:30–21:30$, up to $50\text{ kW}$).
   - Overnight base load of $15–18\text{ kW}$.

3. **Time-of-Use Electricity Tariffs**:
   - Off-peak ($00:00–06:00, 23:00–24:00$): $\$0.08/\text{kWh}$.
   - Mid-peak ($06:00–16:00, 21:00–23:00$): $\$0.15–\$0.18/\text{kWh}$.
   - On-peak ($16:00–21:00$): $\$0.32/\text{kWh}$.

4. **EV Arrival Schedules**:
   - 4 to 6 vehicles per day arriving across daytime shifts with realistic dwell times ($3–7\text{ hours}$).
   - Initial SOC uniformly drawn from $15\%$ to $70\%$.
   - Dwell duration verified to ensure reachability of target SOC ($80\%$).

---

## 13. Baseline Algorithms

1. **Immediate Charging (Uncontrolled / Greedy)**:
   - Charges every connected vehicle at maximum power ($7.0\text{ kW}$) immediately upon arrival.
   - Completely ignores solar generation and electricity prices.
   - Causes heavy grid power surges and peak-tariff costs.

2. **Solar-First Charging (Renewable Heuristic)**:
   - Directly couples EV charging to available solar generation ($P_{\text{solar}}$).
   - Falls back to grid power only when vehicle urgency $U \ge 0.75$ threatens departure deficits.

3. **Proposed Method: Tabular Q-Learning**:
   - Learns an adaptive policy balancing solar availability, electricity prices, peak shaving, and deadline urgency.

---

## 14. Performance Metrics & Mathematical Formulas

| # | Metric | Formula | Unit |
| :---: | :--- | :--- | :---: |
| 1 | **Total Grid Electricity Cost** | $\sum_{t=0}^{N-1} P_{\text{grid\_ev}}(t) \cdot \Delta t \cdot c_{\text{grid}}(t)$ | $\$$ |
| 2 | **Total Grid Energy Consumed** | $\sum_{t=0}^{N-1} P_{\text{grid\_ev}}(t) \cdot \Delta t$ | $\text{kWh}$ |
| 3 | **Total Solar Energy Utilized** | $\sum_{t=0}^{N-1} P_{\text{solar\_used}}(t) \cdot \Delta t$ | $\text{kWh}$ |
| 4 | **Solar Contribution Percentage** | $\frac{E_{\text{solar}}}{E_{\text{solar}} + E_{\text{grid}}} \times 100\%$ | $\%$ |
| 5 | **Peak Grid Demand** | $\max_{t} [P_{\text{base\_grid}}(t) + P_{\text{grid\_ev}}(t)]$ | $\text{kW}$ |
| 6 | **Average Departure SOC** | $\frac{1}{M} \sum_{j=1}^{M} \text{SOC}_{j,\text{dep}}$ | $\%$ |
| 7 | **Target SOC Achievement Rate** | $\frac{1}{M} \sum_{j=1}^{M} \mathbf{1}(\text{SOC}_{j,\text{dep}} \ge 79.9\%) \times 100\%$ | $\%$ |
| 8 | **Total Unmet Energy** | $\sum_{j=1}^{M} \max\left(0, \frac{80 - \text{SOC}_{j,\text{dep}}}{100}\right) \cdot C_{\text{battery}}$ | $\text{kWh}$ |
| 9 | **Total EV Charging Energy** | $E_{\text{grid}} + E_{\text{solar}}$ | $\text{kWh}$ |

---

## 15. Experimental Results & Visualizations

### Summary Benchmark on 20 Independent Test Episodes

| Strategy | Grid Cost ($) | Solar Energy (kWh) | Solar Share (%) | Peak Demand (kW) | Avg Departure SOC (%) | Target Achieved (%) | Unmet Energy (kWh) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Immediate Charging** | \$7.86 ± 3.82 | 51.41 ± 19.29 | 64.09% | 60.16 ± 2.63 | 82.25% | 100.0% | 0.00 kWh |
| **Solar-First Charging** | \$5.98 ± 3.61 | 52.86 ± 18.39 | 67.79% | 58.75 ± 4.78 | 80.98% | 71.25% | 0.59 kWh |
| **RL (Q-Learning)** | **\$7.20 ± 3.53** | **50.72 ± 19.62** | **63.87%** | **58.83 ± 4.35** | **81.56%** | **94.75%** | **0.63 kWh** |

### Key Academic Observations & Pareto Trade-Off Analysis
1. **The Cost vs. Service Satisfaction Trade-off**:
   - While the heuristic **Solar-First Charging** achieves the lowest grid electricity expenditure (\$5.98), it does so by strictly withholding charging during overcast or zero-solar periods. Consequently, **nearly 29% of EVs fail to achieve their target departure SOC** (only 71.25% target satisfaction rate, with 0.59 kWh average deficit per vehicle).
   - In contrast, **Immediate Charging** achieves 100% target satisfaction, but incurs the highest operating expense (\$7.86) and creates the highest peak grid loading (60.16 kW), exacerbating distribution transformer stress during evening on-peak periods.
   - **The proposed Tabular Q-Learning Agent** autonomously discovers a superior Pareto compromise: it achieves **94.75% target SOC satisfaction** (averaging 81.56% departure SOC), reduces peak substation demand to **58.83 kW**, and lowers electricity procurement cost to **\$7.20** by opportunistically charging during low-tariff hours and clean solar generation windows.

2. **Substation Peak Shaving**:
   - Peak grid power demand is reduced from 60.16 kW (Immediate) down to 58.83 kW (RL), providing essential congestion relief to the upstream distribution grid.

3. **Hyperparameter Sensitivity ($\alpha$)**:
   - $\alpha = 0.10$ achieves the highest training convergence reward (54.87) and balanced evaluation performance (92.08% satisfaction, 0.41 kWh unmet).
   - $\alpha = 0.01$ learns too slowly within the 1200 episodes, resulting in conservative charging and 2.92 kWh unmet energy.
   - $\alpha = 0.50$ learns aggressively, achieving 96.75% target satisfaction with slightly higher operational variance.

4. **Resilience to Environmental Stress (30% Solar Reduction)**:
   - Under reduced solar generation, Solar-First satisfaction plummets to 61.42% (failing nearly 40% of vehicles).
   - The RL agent adapts gracefully, maintaining **93.50% target SOC satisfaction** by dynamically shifting to cost-effective grid charging.

---

## 16. Generated Visualizations

All 14 publication-quality plots are saved to `results/plots/`:
1. `01_solar_generation_vs_time.png`: 24-hour solar generation profile under passing cloud effects.
2. `02_grid_demand_vs_time.png`: Background commercial/residential grid demand profile.
3. `03_electricity_price_vs_time.png`: Dynamic Time-of-Use electricity pricing regimes.
4. `04_ev_arrivals_departures.png`: Fleet dwell windows, arrival/departure times, and SOC targets.
5. `05_training_reward_vs_episodes.png`: Tabular Q-learning convergence and epsilon decay curve.
6. `06_ev_soc_vs_time.png`: EV battery SOC trajectories over the 24-hour simulation.
7. `07_power_allocation_vs_time.png`: Solar vs grid dispatch under the 21 kW station constraint.
8. `08_cost_comparison.png`: Electricity cost comparison across all three strategies.
9. `09_solar_utilization_comparison.png`: Solar energy utilization benchmark.
10. `10_peak_grid_demand_comparison.png`: Substation peak demand reduction.
11. `11_grid_energy_consumption_comparison.png`: Grid energy consumption comparison.
12. `12_departure_soc_satisfaction_comparison.png`: Departure SOC and target achievement rates.
13. `13_learning_rate_experiment_comparison.png`: Convergence dynamics under $\alpha \in \{0.01, 0.1, 0.5\}$.
14. `14_solar_sensitivity_analysis.png`: System resilience under a 30% reduction in solar availability.

---

## 17. Limitations & Future Work

### Limitations
- **Discretization Granularity**: Tabular Q-learning requires discrete state bins; very fine bins induce the curse of dimensionality.
- **Fixed Topology**: Currently modeled for a single charging depot with up to 3 ports.

### Future Work
- Extension to continuous action spaces using Deep RL (DDPG, PPO, or SAC).
- Vehicle-to-Grid (V2G) bi-directional power flow injection.
- Station battery energy storage systems (BESS) co-located with PV arrays.

---

## 18. Installation & Execution Instructions

### Prerequisites
- Python 3.9+ (Tested on Python 3.13)
- `numpy`, `pandas`, `matplotlib`, `pytest`

### Setup
```bash
pip install -r requirements.txt
```

### Execution Commands
```bash
# 1. Generate synthetic datasets and environmental profiles
python main.py --mode generate

# 2. Train Tabular Q-Learning agent
python main.py --mode train

# 3. Evaluate trained agent and baselines on 20 test episodes
python main.py --mode evaluate

# 4. Run parameter (learning rate) and solar sensitivity experiments
python main.py --mode experiments

# 5. Run the complete pipeline end-to-end
python main.py --mode all
```

### Running Automated Tests
```bash
pytest tests/
```

