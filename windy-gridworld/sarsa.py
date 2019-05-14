import environment as env
from algorithm import Algorithm

VERSION_NUMBER = 1.0


class Sarsa(Algorithm):
    """Implement a Sarsa algorithm for the Windy Gridworld Environment."""

    def step(self, observation, reward):
        """Perform a complete step of the algorithm."""
        # Computes the coordinates for the current state
        prev_s, prev_a = self.get_position(self.action)

        # Replace the current state with the new state
        self.state = observation

        # Chooses action to perform based on new state
        self.action = self.select_action()

        # Computes the coordinates for the new state
        new_s, new_a = self.get_position(self.action)

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
