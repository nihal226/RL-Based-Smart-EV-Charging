"""Main CLI entry point for the RL-Based Smart EV Charging System.
Supports modular execution via:
    python main.py --mode generate
    python main.py --mode train
    python main.py --mode evaluate
    python main.py --mode experiments
    python main.py --mode all
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd

from config import (
    TRAIN_DATA_DIR,
    TEST_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    TRAINING_EPISODES,
)
from experiments.generate_data import generate_all_datasets
from experiments.train import run_training
from experiments.evaluate import run_full_evaluation, evaluate_policy_on_episode
from experiments.parameter_analysis import run_learning_rate_experiments
from experiments.solar_sensitivity import run_solar_sensitivity_analysis

from visualization.environment_plots import generate_all_environment_plots
from visualization.training_plots import (
    plot_training_rewards,
    plot_soc_trajectories,
    plot_power_allocation,
)
from visualization.comparison_plots import (
    plot_cost_comparison,
    plot_solar_utilization_comparison,
    plot_peak_demand_comparison,
    plot_grid_energy_consumption_comparison,
    plot_departure_soc_satisfaction_comparison,
    plot_learning_rate_comparison,
    plot_solar_sensitivity,
)
from environment.ev_environment import EVChargingEnv
from agents.q_learning import QLearningAgent


def mode_generate():
    """Generates synthetic datasets and baseline environment profile visualizations."""
    print("\n" + "="*70)
    print("MODE: GENERATING SYNTHETIC DATASETS & ENVIRONMENTAL PROFILES")
    print("="*70)
    generate_all_datasets(num_train=80, num_test=20)

    # Generate sample environmental visualizations (Graphs 1 to 4)
    sample_prof = pd.read_csv(TRAIN_DATA_DIR / "episode_000_profile.csv")
    sample_evs = pd.read_csv(TRAIN_DATA_DIR / "episode_000_evs.csv")
    generate_all_environment_plots(sample_prof, sample_evs)
    print("[Success] Data generation and environmental plots complete.")


def mode_train(episodes: int = TRAINING_EPISODES):
    """Trains the Tabular Q-Learning agent and plots the training learning curve."""
    print("\n" + "="*70)
    print(f"MODE: TRAINING TABULAR Q-LEARNING AGENT ({episodes} EPISODES)")
    print("="*70)
    agent, history_df = run_training(num_episodes=episodes)

    # Generate Graph 5: RL convergence and epsilon decay
    plot_training_rewards(history_df)
    print("[Success] Agent training and convergence plot complete.")


def mode_evaluate():
    """Evaluates RL policy and baselines on 20 independent test episodes, generating Graphs 6-12."""
    print("\n" + "="*70)
    print("MODE: BENCHMARKING & MULTI-EPISODE EVALUATION (20 TEST EPISODES)")
    print("="*70)
    detailed_df, summary_df = run_full_evaluation()

    # Generate sample test episode operational profile for Graphs 6 & 7
    model_path = MODELS_DIR / "trained_q_table.pkl"
    agent = QLearningAgent(num_states=1296, num_actions=3)
    agent.load(model_path)

    test_prof = TEST_DATA_DIR / "episode_000_profile.csv"
    test_ev = TEST_DATA_DIR / "episode_000_evs.csv"
    env = EVChargingEnv()
    env.load_episode_from_csv(test_prof, test_ev)
    _, history = evaluate_policy_on_episode(env, "RL (Q-Learning)", agent)

    # Graph 6: EV Battery SOC vs time
    plot_soc_trajectories(history, env.evs)
    # Graph 7: Power allocation vs time
    plot_power_allocation(history)

    # Graphs 8 to 12: Strategy Comparisons
    plot_cost_comparison(detailed_df)
    plot_solar_utilization_comparison(detailed_df)
    plot_peak_demand_comparison(detailed_df)
    plot_grid_energy_consumption_comparison(detailed_df)
    plot_departure_soc_satisfaction_comparison(detailed_df)
    print("[Success] Multi-episode evaluation and comparison plots complete.")


def mode_experiments():
    """Runs parameter sensitivity experiments (learning rate & solar availability)."""
    print("\n" + "="*70)
    print("MODE: PARAMETER EXPERIMENTS & SENSITIVITY ANALYSIS")
    print("="*70)
    # Learning rate experiments
    run_learning_rate_experiments(episodes_per_exp=1200)
    lr_history_df = pd.read_csv(METRICS_DIR / "lr_experiments_history.csv")
    plot_learning_rate_comparison(lr_history_df)

    # Solar sensitivity experiments
    sensitivity_df = run_solar_sensitivity_analysis()
    plot_solar_sensitivity(sensitivity_df)
    print("[Success] Parameter experiments and sensitivity plots complete.")


def mode_all():
    """Executes the complete end-to-end pipeline."""
    print("\n" + "#"*70)
    print("# RUNNING COMPLETE END-TO-END PIPELINE")
    print("#"*70)
    mode_generate()
    mode_train()
    mode_evaluate()
    mode_experiments()
    print("\n" + "#"*70)
    print("# COMPLETE PIPELINE EXECUTION FINISHED SUCCESSFULLY")
    print("#"*70)


def main():
    parser = argparse.ArgumentParser(description="RL-Based Smart EV Charging System")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["generate", "train", "evaluate", "experiments", "all"],
        help="Pipeline execution mode (default: all)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=TRAINING_EPISODES,
        help=f"Number of training episodes (default: {TRAINING_EPISODES})",
    )

    args = parser.parse_args()

    if args.mode == "generate":
        mode_generate()
    elif args.mode == "train":
        mode_train(episodes=args.episodes)
    elif args.mode == "evaluate":
        mode_evaluate()
    elif args.mode == "experiments":
        mode_experiments()
    elif args.mode == "all":
        mode_all()


if __name__ == "__main__":
    main()

