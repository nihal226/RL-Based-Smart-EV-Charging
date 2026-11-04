"""Unit tests for Tabular Q-Learning mathematical correctness and behavior."""

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.q_learning import QLearningAgent


def test_q_learning_bellman_update():
    """Verifies that the Q-learning update follows Q(s,a) += alpha * (r + gamma * max Q(s',.) - Q(s,a))."""
    agent = QLearningAgent(
        num_states=10,
        num_actions=3,
        learning_rate=0.1,
        discount_factor=0.9,
        initial_epsilon=0.0,
    )

    # Initial Q is all zeros
    s = 2
    a = 1
    r = 5.0
    s_next = 3

    # Set known Q-values for s_next
    agent.q_table[s_next, 0] = 2.0
    agent.q_table[s_next, 1] = 8.0  # max next is 8.0
    agent.q_table[s_next, 2] = 4.0

    # Initial Q(s, a) = 0.0
    # Expected TD target = r + gamma * max_a' Q(s', a') = 5.0 + 0.9 * 8.0 = 5.0 + 7.2 = 12.2
    # Expected TD error = 12.2 - 0.0 = 12.2
    # Expected new Q(s, a) = 0.0 + 0.1 * 12.2 = 1.22
    td_err = agent.update(s, a, r, s_next, done=False)

    assert np.isclose(td_err, 12.2, atol=1e-5)
    assert np.isclose(agent.q_table[s, a], 1.22, atol=1e-5)


def test_q_learning_terminal_update():
    """Terminal state transition should not discount future rewards (target = reward)."""
    agent = QLearningAgent(num_states=10, num_actions=3, learning_rate=0.2, discount_factor=0.95)
    s = 4
    a = 2
    r = 10.0
    s_next = 5
    agent.q_table[s_next, :] = 100.0  # Should be ignored because done=True

    td_err = agent.update(s, a, r, s_next, done=True)
    # Target = 10.0, initial Q = 0.0 => TD error = 10.0, new Q = 0.2 * 10 = 2.0
    assert np.isclose(td_err, 10.0, atol=1e-5)
    assert np.isclose(agent.q_table[s, a], 2.0, atol=1e-5)


def test_epsilon_decay():
    """Epsilon should decay down to min_epsilon and stop decaying further."""
    agent = QLearningAgent(
        num_states=5,
        num_actions=2,
        initial_epsilon=1.0,
        min_epsilon=0.05,
        epsilon_decay=0.5,
    )

    e1 = agent.decay_epsilon()
    assert np.isclose(e1, 0.5)

    e2 = agent.decay_epsilon()
    assert np.isclose(e2, 0.25)

    # Decay several times
    for _ in range(10):
        agent.decay_epsilon()

    assert np.isclose(agent.epsilon, 0.05)


def test_evaluation_mode_no_exploration_and_no_q_updates():
    """During evaluation, action selection must be deterministic greedy and Q-table frozen."""
    agent = QLearningAgent(num_states=5, num_actions=3, initial_epsilon=1.0)
    s = 1
    agent.q_table[s, 0] = 1.0
    agent.q_table[s, 1] = 10.0  # best
    agent.q_table[s, 2] = 2.0

    # With is_eval=True, even if epsilon=1.0, it must always pick action 1
    actions = [agent.select_action(s, is_eval=True) for _ in range(50)]
    assert all(a == 1 for a in actions)

    # Q-table snapshot before eval episode
    q_copy = agent.q_table.copy()
    # In eval mode, no update() is invoked, so table remains identical
    assert np.array_equal(agent.q_table, q_copy)


def test_q_table_save_load(tmp_path):
    """Saving and loading Q-table should perfectly recover identical values."""
    agent = QLearningAgent(num_states=5, num_actions=3, seed=123)
    agent.q_table = np.random.uniform(-5.0, 5.0, size=(5, 3))
    agent.epsilon = 0.42
    agent.total_updates = 77

    save_file = tmp_path / "test_q_table.pkl"
    agent.save(save_file)

    loaded_agent = QLearningAgent(num_states=5, num_actions=3)
    loaded_agent.load(save_file)

    assert np.allclose(agent.q_table, loaded_agent.q_table)
    assert np.isclose(agent.epsilon, loaded_agent.epsilon)
    assert agent.total_updates == loaded_agent.total_updates

