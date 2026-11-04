"""Tabular Q-Learning Agent implementation for Smart EV Charging.
Adheres strictly to the Bellman optimality update equation:
    Q(s, a) = Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
"""

from pathlib import Path
from typing import Tuple, Union, Optional
import pickle
import numpy as np

from config import (
    LEARNING_RATE,
    DISCOUNT_FACTOR,
    INITIAL_EPSILON,
    MIN_EPSILON,
    EPSILON_DECAY,
    ACTIONS,
)


class QLearningAgent:
    """Tabular Q-Learning agent with epsilon-greedy exploration and serialization."""

    def __init__(
        self,
        num_states: int = 1296,
        num_actions: int = len(ACTIONS),
        learning_rate: float = LEARNING_RATE,
        discount_factor: float = DISCOUNT_FACTOR,
        initial_epsilon: float = INITIAL_EPSILON,
        min_epsilon: float = MIN_EPSILON,
        epsilon_decay: float = EPSILON_DECAY,
        seed: Optional[int] = None,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng(seed)
        # Initialize Q-table with zeros (shape: [num_states, num_actions])
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=np.float64)

        # Tracking statistics
        self.total_updates = 0

    def select_action(self, state_idx: int, is_eval: bool = False) -> int:
        """Selects an action using epsilon-greedy exploration.

        Parameters:
        -----------
        state_idx : int
            Linear index of the current discrete state.
        is_eval : bool
            If True, exploration is completely disabled (epsilon = 0.0, strictly greedy).

        Returns:
        --------
        int: Chosen discrete action index (0, 1, or 2).
        """
        if not is_eval and self.rng.uniform(0.0, 1.0) < self.epsilon:
            # Explore: uniform random action
            return int(self.rng.choice(self.num_actions))

        # Exploit: choose greedy action with random tie-breaking
        q_values = self.q_table[state_idx]
        max_q = np.max(q_values)
        best_actions = np.where(np.isclose(q_values, max_q, atol=1e-8))[0]
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state_idx: int,
        action_idx: int,
        reward: float,
        next_state_idx: int,
        done: bool,
    ) -> float:
        """Performs standard Tabular Q-learning update.

        Formula:
            TD_Target = reward + (0 if done else gamma * max_a' Q(s', a'))
            TD_Error  = TD_Target - Q(s, a)
            Q(s, a)  += alpha * TD_Error

        Returns:
        --------
        float: TD error for logging and diagnostic tracking.
        """
        current_q = self.q_table[state_idx, action_idx]

        if done:
            target_q = reward
        else:
            best_next_q = np.max(self.q_table[next_state_idx])
            target_q = reward + self.gamma * best_next_q

        td_error = target_q - current_q
        self.q_table[state_idx, action_idx] += self.learning_rate * td_error
        self.total_updates += 1

        return float(td_error)

    def decay_epsilon(self) -> float:
        """Applies episode-level exponential decay to epsilon."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves the trained Q-table and metadata to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q_table": self.q_table,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "total_updates": self.total_updates,
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)

    def load(self, filepath: Union[str, Path]) -> None:
        """Loads a previously saved Q-table from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        with open(filepath, "rb") as f:
            payload = pickle.load(f)
        self.q_table = payload["q_table"]
        self.learning_rate = payload.get("learning_rate", self.learning_rate)
        self.gamma = payload.get("gamma", self.gamma)
        self.epsilon = payload.get("epsilon", self.epsilon)
        self.total_updates = payload.get("total_updates", 0)

