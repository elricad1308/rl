import pickle
import random
import numpy as np
import environment as env

VERSION_NUMBER = 1.0


class Algorithm(object):
    """Implement a Q-Learning algorithm for Windy Gridworld environment."""

    def __init__(self, alpha, epsilon, gamma, king=False, stochastic=False):
        """Create a new agent."""
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        # If 'king', then are 9 actions available, otherwise 5
        self.n_actions = 9 if king else 5

        # As wind is configurable, we need to store the 'max' wind
        # Also, if stochastic, max wind speed is 1 higher
        self.max_wind = np.amax(env.WIND) + 1
        if stochastic:
            self.max_wind += 1

        # Initialize the Q estimate
        self.q = np.zeros((
          env.WORLD_W * env.WORLD_H,
          self.n_actions * self.max_wind
        ))

        # Initialize first step
        self.state = None

        # Initialize the current action
        self.action = 0

        # Iteration count
        self.iteration = 0

    def compute_position(self, action):
        """Compute the position of 'next' state if 'action' is applied.

        Args:
          - action (int): the action to perform.

        """
        x = self.state[0]
        y = self.state[1]

        # Applies the action to the next position of the agent
        if action == env.ACTION_NORTH:
            y -= 1

        if action == env.ACTION_NORTH_EAST:
            x += 1
            y -= 1

        if action == env.ACTION_EAST:
            x += 1

        if action == env.ACTION_SOUTH_EAST:
            x += 1
            y += 1

        if action == env.ACTION_SOUTH:
            y += 1

        if action == env.ACTION_SOUTH_WEST:
            x -= 1
            y += 1

        if action == env.ACTION_WEST:
            x -= 1

        if action == env.ACTION_NORTH_WEST:
            x -= 1
            y -= 1

        # Keeps the coordinates within the grid
        x = 0 if x < 0 else x
        x = env.WORLD_W - 1 if x >= env.WORLD_W else x
        y = 0 if y < 0 else y
        y = env.WORLD_H - 1 if y >= env.WORLD_H else y

        return (x, y)

    def debug(self):
        """Print information about the state of the agent."""
        message = (
          f"Iteration: {self.iteration}\t"
          f"Time steps: {self.time_step}\t"
          f"Total reward: {self.total_reward}"
        )

        print(message)

    def load(self, filename):
        """Load an agent's state from filename."""
        data = pickle.load(open(filename, "rb"))

        data_version = data[0]

        if data_version == VERSION_NUMBER:
            self.q = data[1]
            self.n_actions = data[2]
            self.max_wind = data[3]
            self.iteration = data[4]
        else:
            print(
              f"ERROR: Data version is {data_version}, and therefore "
              f"cannot be loaded (current version: {VERSION_NUMBER})"
            )

    def reset(self, obs):
        """Reset the algorithm to initial configuration.

        The initial state S is set to the given observation. Based
        on S, an action A is selected and returned.

        Args:
          - obs (numpy.array): the initial observation from the
              environment.

        Returns:
          (int): the action selected to perform based on the initial
            observation.

        """
        # Sets initial state
        self.state = obs

        # Select an action based on initial state
        self.action = self.select_action()

        # Increase iteration count
        self.iteration += 1

        # Reset time step count
        self.time_step = 0

        # Reset total reward count
        self.total_reward = 0

        return self.action

    def save(self, filename):
        """Save the state of the agent to outputfile."""
        data = [
          VERSION_NUMBER,
          self.q,
          self.n_actions,
          self.max_wind,
          self.iteration
        ]

        pickle.dump(data, open(filename, "wb"))

    def select_action(self):
        """Select an action based on current state.

        With probability 1 - epsilon, the greedy action (that is, the
        one with highest estimate q) is selected. With the remaining
        probability epsilon, a random action is selected.

        Returns:
          (int): the action selected for the current state, based on
            an epsilon greedy policy.

        """
        # X coordinate of the q estimate is the x coordinate (state[0])
        # multiplied by the y coordinate (state[1])
        x_coord = (self.state[0] * self.state[1]) + self.state[1]

        # Y coordinate of the q estimate is the wind strength (state[2])
        # multiplied by the number of actions
        y_coord = self.n_actions * self.state[2]

        # With probability 1 - epsilon, greedy action is selected
        if random.random() < 1.0 - self.epsilon:
            # Search the greedy action for current state
            greedy_action = 1
            max_estimate = -(2 ** 32)

            for a in range(1, self.n_actions):
                estimate = self.q[x_coord][y_coord + a]

                # If the estimate is higher, it is always selected
                if estimate > max_estimate:
                    greedy_action = a
                    max_estimate = estimate
                # elif estimate == max_estimate:
                #    if self.state[0] <= env.GOAL_X + 1:
                #        greedy_action = env.ACTION_EAST
                #    elif self.state[1] <= (env.GOAL_Y) and self.state[2] == 0:
                #        greedy_action = env.ACTION_SOUTH
                #    elif self.state[0] > env.GOAL_X:
                #        greedy_action = env.ACTION_WEST
                # If the estimate is equal, is replaced if it will
                # make the target closest to the goal
                elif estimate == max_estimate:
                    nxt_a = self.compute_position(a)
                    nxt_g = self.compute_position(greedy_action)
                    g_x = env.GOAL_X
                    g_y = env.GOAL_Y

                    dist_a = abs(nxt_a[0] - g_x) + abs(nxt_a[1] - g_y)
                    dist_g = abs(nxt_g[0] - g_x) + abs(nxt_g[1] - g_y)

                    greedy_action = a if dist_a < dist_g else greedy_action

            action = greedy_action
        # With probability epsilon, a random action is selected
        else:
            action = random.randrange(1, self.n_actions)

        return action

    def step(self, observation, reward):
        """Perform a complete step of the algorithm."""
        # Computes the coordinates for the current state
        prev_s = (self.state[0] * self.state[1]) + self.state[1]
        prev_a = (self.state[2] * self.n_actions) + self.action

        # Replace the current state with the new state
        self.state = observation

        # Chooses action to perform based on new state
        self.action = self.select_action()

        # Computes the coordinates for the new state
        new_s = (self.state[0] * self.state[1]) + self.state[1]

        # Finds the action 'a' with maximum estimate on new state
        max_a = 0
        max_estimate = -(2 ** 32)
        for a in range(1, self.n_actions):
            new_a = (self.state[2] * self.n_actions) + a
            estimate = self.q[new_s][new_a]

            if estimate > max_estimate:
                max_a = a
                max_estimate = estimate

        new_a = (self.state[2] * self.n_actions) + max_a

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
