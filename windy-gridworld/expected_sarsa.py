
from algorithm import Algorithm

VERSION_NUMBER = 1.0


class ExpectedSarsa(Algorithm):
    """Implement an Expected Sarsa algorithm for Windy Gridworld."""

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

        # Find the 'greedy' action
        for a in range(1, self.n_actions):
            estimate = self.q[new_s][new_a + a]

            if estimate > max_estimate:
                max_a = a
                max_estimate = estimate

        # Computes the expectaction for the rule update
        p_ngreedy = self.epsilon / (self.n_actions - 1)
        expectation = 0

        for a in range(1, self.n_actions):
            # P of choosing greedy action is 1 - e + e/|A|
            if a == max_a:
                p = 1.0 - self.epsilon + p_ngreedy
            # P of choosing non greedy actions is e/|A|
            else:
                p = p_ngreedy

            expectation += (p * self.q[new_s][new_a + a])

        # Performs the estimate update
        prev_e = self.q[prev_s][prev_a]
        error = self.alpha * (reward + (self.gamma * expectation) - prev_e)

        self.q[prev_s][prev_a] = prev_e + error

        # Stores total reward
        self.total_reward += reward

        # Increase time step count
        self.time_step += 1

        # Sends the new action to the environment
        return self.action
