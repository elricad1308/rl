import pickle
import random
import numpy as np

LUNAR_LANDING_ACTIONS = 4
"""Number of actions available on the Lunar Landing environment."""

PRECISION = 2
"""Desired precision for the states to preserve (decimal places)."""

VERSION_NUMBER = 1.0
"""Current version of the algorithm."""


class Algorithm(object):
    """Implement an On-Policy Monte Carlo agent for Lunar Landing.

    Attributes:
      - actions (list): a list that contains the actions selected on
          the current episode.

      - episode (int): the number of performed episodes.

      - epsilon (float): a small value in [0, 1) used for the epsilon
          greedy policy selection.

      - gamma (float): a small value in [0, 1) used as the discount
          rate for the return.

      - policy (dict): the policy of the agent. It contains as keys the
          IDs of all known states, and as values an integer in [0, 3]
          which represents the greedy action to select for that state.

      - q (dict): the action-value function of the agent. It contains
          as keys the IDs of all known states, and as values a numpy
          array of 4 elements, which contains the values of each action
          for that state.

      - rewards (list): a list that contains the sequence of all
          rewards received on the current episode.

      - seen (int): number of previously known states visited on the
          current episode.

      - states (dict): the dictionary containing ALL known states. It
          maps from states as received from the observation (in the
          case of Lunar Landing, an 8-element array) to a natural int
          used as the State ID.

      - success (int): the number of successful episodes (episodes with
          positive total return)

      - test_mode (bool): a flag that indicates if the agent runs in
          test mode or not. In test mode, no new states are registered,
          and no policy iteration is done at the end of each episode.
          Test mode is designed to run the simulation using the target
          policy as fast as possible, without the additional burden of
          all the steps required for policy iteration.

      - total_reward (float): the total reward obtainer by the agent
          during the current episode.

      - unseen (int): list of previously unknown states visited on the
          current episode.

      - visited (list): a list that contains the IDs of the visited
          states on the current episode.

    """

    def __init__(self, epsilon, gamma, test=False):
        """Create a new agent.

        Args:
          - epsilon (float): the value to use for the epsilon-greedy
              policy selection.

          - gamma (float): the value to use as the discount rate for
              the return.

          - [test] (bool): a flag that indicates if the agent runs in
              simulation mode or not.

        """
        # Episode count for the agent
        self.episode = 1

        # Epsilon attribute for the epsilon-greedy policy.
        self.epsilon = epsilon

        # Gamma attribute for the return formula.
        self.gamma = gamma

        # Dictionary containing the amount of times each pair state
        # and action is selected.
        self.n = dict()

        # Dictionary containing the policy. It contains as keys all
        # seen states IDs, and as values de greedy action for that
        # state
        self.policy = dict()

        # Dictionary containing the estimate of the action-state value
        # function.
        self.q = dict()

        # Dictionary containing ALL seen states. It functions as a one
        # to one mapping from observed 'states' to an assigned ID, in
        # order to make structures more efficient.
        self.states = dict()

        # Cantidad de episodios 'exitosos' (con retorno positivo)
        self.success = 0

        # Flag for the test mode
        self.test_mode = test

        # Prepares the agent for a new episode.
        self.reset()

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
        episode_size = len(self.visited)

        perc_seen = (self.seen / episode_size) * 100
        perc_unseen = (self.unseen / episode_size) * 100

        print(
          f"Episode: {self.episode}\t"
          f"Total reward: {self.total_reward:.2f}\t"
          f"Success: {self.success}\t"
          f"Ep. size: {episode_size}\t"
          f"({perc_seen:.2f}% seen / {perc_unseen:.2f}% unseen)\t"
          f"States: {len(self.states)}"
        )

    def load(self, filename):
        """Load a saved agent.

        Agents can be saved to preserve their state, and the loaded
        again using this method. The given 'filename' should contains
        a valid agent, which is represented as a pickled list.

        The first element of the agent's list is the version number:
        if this value differs from the current version, then the agent
        cannot be loaded.

        If the saved agent's version is compatible with the current
        version, then all the data structures stored on the 'filename'
        are used to restore the agent.

        Args:
          - filename (string): the file that contains the agent.

        """
        data = pickle.load(open(filename, "rb"))

        data_version = data[0]

        if data_version == VERSION_NUMBER:
            # Episode number and number of success
            self.episode = data[1]
            self.success = data[2]

            # Target policy
            self.policy = data[3]

            # Action-value estimate Q
            self.q = data[4]

            # Cumulative weight sum C
            self.n = data[5]

            # Dict with the known states
            self.states = data[6]

            # Prepare the agent for a new episode
            self.reset()
        else:
            print(
              f"ERROR: Data version is {data_version}, and therefore "
              f"cannot be loaded (current version: {VERSION_NUMBER})"
            )

    def policy_iteration(self, final_reward):
        """Evaluate and update the current policy."""
        # Stores the final reward on the history
        self.rewards.append(final_reward)

        # Increase the episode count
        self.episode += 1

        # Variable that stores the return
        G = 0

        # Maximum time step for this episode
        T = len(self.rewards)

        # Loops the history in reverse order (there is always
        # one extra reward, for that reason we start the
        # computation on T - 2)
        for t in range(T - 2, -1, -1):
            state = self.visited[t]
            action = self.actions[t]
            reward = self.rewards[t + 1]
            n = self.n[state][action]

            # Update the total reward
            self.total_reward += reward

            # Update the return for time t
            G = (self.gamma * G) + reward

            # Updates the q(s, a) estimate
            estimate = self.q[state][action]
            self.q[state][action] = estimate + (1.0 / n) * (G - estimate)

            # Finds the action with max q(s, a)
            max_a = 0
            max_q = self.q[state][0]

            for a in range(LUNAR_LANDING_ACTIONS):
                if(self.q[state][a] > max_q):
                    max_q = self.q[state][a]
                    max_a = a

            # Updates the current policy
            self.policy[state] = max_a

        # A positive total reward means a successful episode
        if self.total_reward > 0:
            self.success += 1

    def reset(self):
        """Prepare the agent for a new episode.

        This method re-creates all the data structures that change with
        each episode (actions, rewards and visited states).

        """
        # List containing actions selected in current episode
        self.actions = list()

        # List containing all rewards received on current episode.
        self.rewards = list()

        # Number of previously seen states in current episode
        self.seen = 0

        # Total reward for current episode
        self.total_reward = 0

        # Number of new episodes created in current episode
        self.unseen = 0

        # List containing states VISITED in current episode
        self.visited = list()

    def save(self, filename):
        """Save the current state of the agent.

        The info saved includes the target policy, the action value
        estimate q, the number of visits to each state-action n, the
        list of known states, the current episode number and the number
        of successful episodes.

        The number of seen and unseen episodes, actions, rewards and
        visited states are not saved, since they are only relevant for
        the current episode. The values of epsilon, gamma and test mode
        neither are saved, since they are not necessary for future
        iterations.

        A list is created with the algorithm version number, episode
        number, amount of successful episodes, policy, action-value
        estimate q, number of visits n and the states (in that order),
        and then the list is pickled on the file with the given
        'filename'.

        Args:
          filename (string): the name of the destiny file.

        """
        data = [
          VERSION_NUMBER,
          self.episode,
          self.success,
          self.policy,
          self.q,
          self.n,
          self.states
        ]

        pickle.dump(data, open(filename, "wb"))

    def select_action(self, state_id):
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
        # With probability 1 - epsilon, greedy action is chosed
        if random.random() < (1.0 - self.epsilon):
            action = 0
            greedy_estimate = self.q[state_id][0]

            for a in range(LUNAR_LANDING_ACTIONS):
                if self.q[state_id][a] > greedy_estimate:
                    greedy_estimate = self.q[state_id][a]
                    action = a

        # With probability epsilon, a random action is chosed
        else:
            action = random.randrange(0, LUNAR_LANDING_ACTIONS)

        return action

    def step(self, obs, reward=None):
        """Perform an entire time step of the agent.

        First, the observation sent from the environment is 'hashed'
        and transformed into a state.

        Then, the visit to the state is registered, and an action is
        selected by the policy for the agent to perform.

        After that, if a reward is given, it is stored on the rewards
        history for this episode.

        Args:
          obs (numpy.array): the observation sent by the enviroment.

          [reward] (float): Optional. The reward sent from the
            environment. Defaults to None.

        """
        state = self.create_state(obs)

        # action = self.visit_state(state)
        action = self.visit_state(state)

        # A reward of None indicates that this is the first time step.
        if reward is not None:
            self.rewards.append(reward)
        # First time step has a reward of zero
        else:
            self.rewards.append(0)

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
            # Selects a random action as the greedy action for the
            # new state
            action = random.randrange(0, LUNAR_LANDING_ACTIONS)

            # Registers the state as unseen
            self.unseen += 1

            if not self.test_mode:
                # Assign the next available State ID
                state_id = len(self.states)

                # Registers the new state on the dict
                self.states[state] = state_id

                # Stores the random action as the greedy action on the
                # target policy
                self.policy[state_id] = action

                # Create the array for the state on the q action value
                # estimate
                self.q[state_id] = np.zeros(LUNAR_LANDING_ACTIONS)

                # We also create the array to store the amount of times the
                # state and each action are selected
                self.n[state_id] = np.zeros(LUNAR_LANDING_ACTIONS)
        # Otherwise, recovers the state_id for that state, and gets a
        # epsilon-greedy action for it
        else:
            state_id = self.states[state]

            # Selects an action according to policy
            action = self.select_action(state_id)

            # Registers the state as seen
            self.seen += 1

        # Registers only occur if not in test mode
        if not self.test_mode:
            # Adds the state_id to the episode's history
            self.visited.append(state_id)

            # Adds the selected action to the episode's history
            self.actions.append(action)

            # Increase the number of times pair state-action is selected
            self.n[state_id][action] += 1

        return action
