import pickle
import random
import numpy as np

LUNAR_LANDING_ACTIONS = 4
"""Number of actions available on the Lunar Landing environment."""

PRECISION = 1
"""Desired precision for the states to preserve (decimal places)."""

VERSION_NUMBER = 1.0
"""Current version of the algorithm."""


class Algorithm(object):
    """Implement a Q-Learning agent for Lunar Landing."""

    def __init__(self, alpha, epsilon, gamma, test):
        """Create a new agent."""
        # Episode count for the agent
        self.episode = 1

        # Alpha attribute for the update rule
        self.alpha = alpha

        # Epsilon attribute for the epsilon-greedy policy.
        self.epsilon = epsilon

        # Gamma attribute for the return formula.
        self.gamma = gamma

        # Dictionary containing the estimate of the action-state value
        # function.
        self.q = dict()

        # Dictionary containing ALL seen states. It functions as a one
        # to one mapping from observed 'states' to an assigned ID, in
        # order to make structures more efficient.
        self.states = dict()

        # Cantidad de episodios 'exitosos' (con retorno diferente del minimo)
        self.success = 0

        # Flag for the test mode
        self.test_mode = test

    def create_state(self, obs):
        """Transform an observation into a state.

        The Lunar Landing environment provides at each time step an 8
        element numpy array, containing information such as the X and
        Y coordinates of the pad (which are continuous, and are scaled
        to the viewport), the speed, angle and angular speed of the
        pad, and flags that indicate whether the pad has made contact
        with the floor or not.

        Due to all values being real, the amount of possible states is
        overwhelming, and thus, in order to maintain efficiency, the
        values from the observation array are transformed following a
        series of rules based on the 'heuristic' present on the Lunar
        Landing example.

        Args:
          - obs (numpy.array) the observation array, as sent from the
              environment.

        Returns:
          (string) a 'hash' string of the state, created by truncating
            the values and joining them together.

        """
        # Pad angle should point to center. To compute it, we use the
        # horizontal coordinate (obs[0]) and the horizontal speed
        # (obs[2])
        target_angle = obs[0] * 0.5 + obs[2] * 1.0

        # Any value higher than 0.4 radians is equally bad, and thus
        # are all the same
        if target_angle > 0.4:
            target_angle = 0.4
        if target_angle < -0.4:
            target_angle = -0.4

        # To compute the fix to perform on the angle, we use the target
        # angle, the current angle (obs[4]) and the angle speed
        # (obs[5])
        todo_angle = (target_angle - obs[4]) * 0.5 - (obs[5]) * 1.0

        # The target y should be proportional to the horizontal offset
        target_hover = 0.55 * np.abs(obs[0])

        # To compute the fix to perform on the hover, we use the target
        # hover, the vertical coordinate (obs[1]) and the vertical
        # speed (obs[3])
        todo_hover = (target_hover - obs[1]) * 0.5 - (obs[3]) * 0.5

        # If any leg has made contact, we should NOT modify the angle,
        # because it will cause the pad to lose stability. We should
        # only focus on reduce vertical speed, to make the landing as
        # smooth as possible. Leg contact is indicated by either obs[6]
        # (left lef) or obs[7] (right leg).
        if obs[6] or obs[7]:
            todo_angle = 0
            todo_hover = -(obs[3]) * 0.5

        # All the information we need is thus the targets angle and
        # hover, and the todo angle and hover. This values are stored
        # on a list and joined together on a string to create the hash
        state = [
          str(round(target_angle, PRECISION)),
          str(round(target_hover, PRECISION)),
          str(round(todo_angle, PRECISION)),
          str(round(todo_hover, PRECISION))
        ]

        return ' '.join(state)

    def debug(self):
        """Print debug information about the state of the agent."""
        perc_seen = (self.seen / self.visited) * 100
        perc_unseen = (self.unseen / self.visited) * 100

        print(
          f"Episode: {self.episode}\t"
          f"Total reward: {self.total_reward:.2f}\t"
          f"Success: {self.success}\t"
          f"Ep. size: {self.visited}\t"
          f"({perc_seen:.2f}% seen / {perc_unseen:.2f}% unseen)\t"
          f"States: {len(self.states)}"
        )

    def load(self, filename):
        """Load agent from filename."""
        data = pickle.load(open(filename, "rb"))

        data_version = data[0]

        if data_version == VERSION_NUMBER:
            self.states = data[1]
            self.q = data[2]
            self.episode = data[3]
            self.success = data[4]
        else:
            print(
              f"ERROR: Data version is {data_version}, and therefore "
              f"cannot be loaded (current version: {VERSION_NUMBER})"
            )

    def policy_iteration(self, reward):
        """Act as a placeholder for the iteration method."""
        self.total_reward += reward

        # Increase the iteration count
        self.episode += 1

        # A positive total reward means a successful episode
        if self.total_reward > 0:
            self.success += 1

    def reset(self, obs):
        """Prepare agent for a new episode."""
        # Previously known states
        self.seen = 0

        # New states found on this episode
        self.unseen = 0

        # Total reward for current episode
        self.total_reward = 0

        # Total number of states on current episode
        self.visited = 1

        # Create state from initial observation
        self.state = self.create_state(obs)

        # Select action from initial state
        self.action = self.visit_state(self.state)

        return self.action

    def save(self, filename):
        """Save agent to filename."""
        data = [
          VERSION_NUMBER,
          self.states,
          self.q,
          self.episode,
          self.success
        ]

        pickle.dump(data, open(filename, "wb"))

    def select_action(self, state):
        """Apply an epsilon-greedy policy to select an action.

        With probability 1 - epsilon, the greedy action for the given
        state_id is selected (based on the value of the estimates).

        With probability epsilon, an action is selected randomly from
        the current policy, using the probabilities stored for each
        action.

        Args:
          state_id (int): the ID of the current state.

        Returns:
          (int): the action to perform on the current state, based on
            an epsilon-soft policy.

        """
        # With probability 1 - epsilon, greedy action is selected
        if random.random() < 1.0 - self.epsilon:
            greedy_action = 0
            greedy_estimate = -(2 ** 32)

            # Find greedy action
            for a in range(LUNAR_LANDING_ACTIONS):
                estimate = self.q[state][a]

                if estimate > greedy_estimate:
                    greedy_action = a
                    greedy_estimate = estimate
        # With probability epsilon, a random action is chosed
        else:
            greedy_action = random.randrange(0, LUNAR_LANDING_ACTIONS)

        return greedy_action

    def step(self, obs, reward):
        """Perform a complete step of the algorithm."""
        # Obtain new state from observation
        state = self.create_state(obs)

        # Select new action for the new state
        action = self.visit_state(state)

        # Update occurs only if not in test mode
        if not self.test_mode:
            # Updates the estimate for previous action
            p_s = self.states[self.state]
            n_s = self.states[state]

            # Search the action with max estimate
            p_estimate = self.q[p_s][self.action]
            n_estimate = -(2 ** 32)
            for a in range(LUNAR_LANDING_ACTIONS):
                estimate = self.q[n_s][a]

                if estimate > n_estimate:
                    n_estimate = estimate

            # Computes the error for the formula
            error = self.alpha * (reward + (self.gamma * n_estimate) - p_estimate)

            self.q[p_s][self.action] = p_estimate + error

        # Replaces the current state and action
        self.state = state
        self.action = action

        # Sums the reward for the episode
        self.total_reward += reward

        return action

    def visit_state(self, state):
        """Register a visit to the given state.

        If the state is not known (that is, if it doesn't exist on the
        'states' dict), then it is registered, and the next available
        state ID is assigned to it. Then, the state is also added to
        the 'policy' dict, with a random action selected as the greedy
        action for the newly registered state. Finally, a new register
        is also added to the 'q' dict, with all action-value estimates
        for the new state set to zero.

        On the other hand, if 'state' is known, then its ID is found on
        the 'states' dict, and the epsilon-greedy policy is applied:
        with probability (1 - e) the greedy action stored on the policy
        for that state is selected, and with probability e a random
        action is selected from [0, 3] (recall that the Lunar Landing
        task has four defined actions).

        Wether the state is known or not, it is added to the 'visited'
        list for this episode, and the selected action is added to the
        'actions' list.

        For return value, the method returns the action to perform.

        Args:
          - state (string): the hash value of the visited state.

        Returns:
          (int): the action selected to perform by the policy of the
            agent.

        """
        # If 'state' has been never seen before, then 'registers' it on
        # the 'states' dictionary, and assigns it a numeric ID.
        if state not in self.states:
            # Assign the next available State ID
            state_id = len(self.states)

            # Selects a random action as the greedy action for the
            # new state
            action = random.randrange(0, LUNAR_LANDING_ACTIONS)

            # Registers the state as unseen
            self.unseen += 1

            if not self.test_mode:
                # Registers the new state on the dict
                self.states[state] = state_id

                # Create the array for the state on the q action value
                # estimate
                self.q[state_id] = np.zeros(LUNAR_LANDING_ACTIONS)
        # Otherwise, recovers the state_id for that state, and gets a
        # epsilon-greedy action for it
        else:
            state_id = self.states[state]

            # Selects an action according to policy
            action = self.select_action(state_id)

            # Registers the state as seen
            self.seen += 1

        self.visited += 1

        return action
