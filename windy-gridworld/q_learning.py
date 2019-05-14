import pickle
import random
import numpy as np
import environment as env

from algorithm import Algorithm

VERSION_NUMBER = 1.0


class QLearning(Algorithm):
    """Implement a Q-Learning algorithm for Windy Gridworld environment."""

    def step(self, observation, reward):
        """Perform a complete step of the algorithm."""
        # Computes the coordinates for the current state
        prev_s, prev_a = self.get_position(self.action)

        # Replace the current state with the new state
        self.state = observation

        # Chooses action to perform based on new state
        self.action = self.select_action()

        # Computes the coordinates for the new state
        new_s, new_a = self.get_position()

        # Finds the action 'a' with maximum estimate on new state
        max_a = 0
        max_estimate = -(2 ** 32)
        for a in range(1, self.n_actions):
            estimate = self.q[new_s][new_a + a]

            if estimate > max_estimate:
                max_a = a
                max_estimate = estimate

        new_s, new_a = self.get_position(max_a)

        # Performs the estimate update
        prev_e = self.q[prev_s][prev_a]
        new_e = self.q[new_s][new_a]
        error = self.alpha * (reward + (self.gamma * new_e) - prev_e)

        self.q[prev_s][prev_a] = prev_e + error

        # Stores total reward
        self.total_reward += reward

        # Increase time step count
        self.time_step += 1

        # Sends the new action to the environment
        return self.action
