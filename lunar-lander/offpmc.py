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
    """Implement an Off-Policy Monte Carlo agent for Lunar Landing.

    Attributes:
      - actions (list): a list that contains the sequence of actions
          selected on the current episode.

      - C (dict): the cumulative sum of the weights to use on the
          weighted importance sampling update rule. It contains as keys
          the IDs of all known states, and as values a 4-element numpy
          array, with each element being the cumulative sum of the
          actions of the agent on that state.

      - episode (int): the number of performed episodes.

      - epsilon (float): a small value in [0, 1) used for the epsilon
          greedy behavior policy.

      - gamma (float): a small value in [0, 1] used as the discount
          rate for the return.

      - policy (dict): the target policy of the agent. It contains as
          keys the ID's of all known states, and as values an integer
          in [0, 3] which represents the action to select for that
          state.

      - Q (dict): the action-value function of the agent's target
          policy. It contains as keys the IDs of all known states, and
          as values a 4-element numpy array, which contains the values
          of each action for that state.

      - rewards (list): a list that contains the sequence of all
          rewards received on the current episode.

      - seen (int): number of previously known states visited on the
          current episode.

      - states (dict): a dictionary that contains ALL known states. It
          maps from states constructed by the observation to a natural
          int used as the State ID.

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

      - visited (list): a list that contains the sequence of the IDs of
          all states visited on the current episode.

    """

    def __init__(self, epsilon, gamma, test=False):
        """Create a new agent.

        Args:
          - epsilon (float): the value to use for the epsilon-greedy
              behavior policy.

          - gamma (float): the value to use as the discount rate for
              the return formula.

          - [test] (bool): a flag that indicates if the agent runs in
              simulation mode or not.

        """
        # Dictionary containing the cumulative sum of the weights for
        # each pair state-action of the agent.
        self.C = dict()

        # Episode count for the agent
        self.episode = 1

        # Epsilon attribute for the behaviour policy.
        self.epsilon = epsilon

        # Gamma attibute for the return formula
        self.gamma = gamma

        # Dictionary containing the target policy.
        self.policy = dict()

        # Dictionary containing the estimate of the action-state value
        # function for the target policy.
        self.Q = dict()

        # Dictionary containing ALL known states.
        self.states = dict()

        # Number of successful episodes (with positive final return)
        self.success = 0

        # Flag for the test mode
        self.test_mode = test

        # Prepare the agent for the first episode.
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
          f"Total reward: {self.total_reward}\t"
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
            self.Q = data[4]

            # Cumulative weight sum C
            self.C = data[5]

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
        # Stores the final reward on the history
        self.rewards.append(final_reward)

        # Increase the episode count
        self.episode += 1

        # Variable that stores the return
        G = 0

        # Variable that stores the cumulative weight sum
        W = 1

        # Maximum time step for this episode
        T = len(self.rewards)

        # Loops the history in reverse order (there is always
        # one extra reward, for that reason we start the
        # computation on T - 2)
        for t in range(T - 2, -1, -1):
            state = self.visited[t]
            action = self.actions[t]
            reward = self.rewards[t + 1]

            # Adds the reward to total
            self.total_reward += reward

            # Discounts the return
            G = (self.gamma * G) + reward

            # Updates the weight
            weight = self.C[state][action] + W
            self.C[state][action] += weight

            # Updates the estimate
            estimate = self.Q[state][action]
            self.Q[state][action] = estimate + ((W / weight) * (G - estimate))

            # Find the greedy action
            greedy_action = 0
            max_estimate = -2 ** 32

            for a in range(LUNAR_LANDING_ACTIONS):
                if self.Q[state][a] > max_estimate:
                    max_estimate = self.Q[state][a]
                    greedy_action = a

            # Updates the target policy with the greedy action
            self.policy[state] = greedy_action

            print(f"A_t = {action} , Greedy = {greedy_action}")

            # If greedy action was not selected, proceed to next episode
            if action != greedy_action:
                break

            # Updates the weight
            W = W * (1.0 / (1.0 - self.epsilon + (self.epsilon / LUNAR_LANDING_ACTIONS)))

        # A positive total reward means a successful landing
        if self.total_reward > 0:
            self.success += 1

    def reset(self):
        """Prepare the agent for a new episode.

        This method re-creates all the data structures that change with
        each episode (the visited states, the selected actions and the
        obtained rewards).

        """
        # List containing the actions selected in the current episode
        self.actions = list()

        # List containing all rewards received in the current episode
        self.rewards = list()

        # Number of previously known states
        self.seen = 0

        # Total reward for current episode
        self.total_reward = 0

        # Number of unknown states
        self.unseen = 0

        # List containing all states visited in current episode
        self.visited = list()

    def save(self, filename):
        """Save the current state of the agent.

        The info saved includes the target policy, the action value
        estimate Q, the cumulative weight sum C, the list of known
        states, the current episode number and the number of successful
        episodes.

        The number of seen and unseen episodes, actions, rewards and
        visited states are not saved, since they are only relevant for
        the current episode. The values of epsilon, gamma and test mode
        neither are saved, since they are not necessary for future
        iterations.

        A list is created with the algorithm version number, episode
        number, amount of successful episodes, policy, action-value
        estimate q, cumulative weight sum and the states (in that
        order), and then the list is pickled on the file with the given
        'filename'.

        Args:
          filename (string): the name of the destiny file.

        """
        data = [
          VERSION_NUMBER,
          self.episode,
          self.success,
          self.policy,
          self.Q,
          self.C,
          self.states
        ]

        pickle.dump(data, open(filename, "wb"))

    def select_action(self, state_id):
        """Apply an epsilon-greedy soft policy to select an action.

        With probability 1 - epsilon, the greedy action (that is, the
        one dictated by the target policy).

        With probability epsilon, a random action is selected, with
        equal probability for each of them.

        This method simulates the behavior policy of the agent.

        Args:
          state_id (int): the ID of the current state.

        Returns:
          (int): the action to perform on that state.
        """
        # In test mode, the target policy is ALWAYS used
        if self.test_mode or random.random() < (1.0 - self.epsilon):
            action = self.policy[state_id]
        else:
            action = random.randrange(0, LUNAR_LANDING_ACTIONS)

        return action

    def step(self, obs, reward=None):
        """Perform an entire time step of the agent.

        First, the observation sent from the environment is 'hashed'
        and transformed into a state.

        Then, the visit to the state is registered, and an action is
        selected by a behavior policy for the agent to perform.

        After that, if a reward is given, it is stored on the rewards
        history for this episode.

        Args:
          obs (numpy.array): the observation sent by the environment.

          [reward] (float): Optional. The reward sent from the
            environment. Defaults to None.

        """
        # Transforms the observation into a state
        state = self.create_state(obs)

        # Selects the action to perform
        action = self.visit_state(state)

        # A reward of None indicates that this is the first time step
        if reward is not None:
            self.rewards.append(reward)
        # Stores a negative reward to use as the zero-index reward
        else:
            self.rewards.append(-1)

        return action

    def visit_state(self, state):
        """Register a visit to the given state.

        If the state is not known (that is, if it doesn't exist on the
        'states' dict), then it is registered with the next available
        state ID, added to the target policy with a random action as
        greedy action for it, and added to the Q action value estimate,
        with all the values for all actions set to zero.

        On the other hand, if 'state' is known, then its ID is found on
        the 'states' dict, and an epsilon-greedy rule is applied to
        simulate the behavior policy: with probability 1 - epsilon, the
        greedy action (that is, the one stored on the target policy) is
        selected, and with epsilon probability one of the four actions
        is selected at random (with equal probability).

        Whether the state is known or not, its ID is added to the
        'visited' list for this episode, and the selected action is
        added to the 'actions' list.

        As return value, the method returns the action to perform.

        Args:
          - state (string): the hash value of the visited state.

        Returns:
          (int): the action selected to perform by the behavior policy
            of the agent.

        """
        # If 'state' is unknown to the agent, registers it
        if state not in self.states:
            # Selects a random action as the greedy action for the
            # new state
            action = random.randrange(0, LUNAR_LANDING_ACTIONS)

            # Counts the episode as 'unseen'
            self.unseen += 1

            # Registers only occur if not in test mode
            if not self.test_mode:
                # Assign the next available State ID
                state_id = len(self.states)

                # Registers the new state on the dict
                self.states[state] = state_id

                # Stores the random action as the greedy action on the
                # target policy
                self.policy[state_id] = action

                # Create the array for the state on the Q action value
                # estimate
                self.Q[state_id] = np.zeros(LUNAR_LANDING_ACTIONS)

                # Create the array for the cumulative weight sum of the
                # state
                self.C[state_id] = np.zeros(LUNAR_LANDING_ACTIONS)
        # Otherwise, recover the state ID, and gets a epsilon-greedy
        # action for it.
        else:
            # Gets the state ID
            state_id = self.states[state]

            # Selects an action according to the behavior policy
            action = self.select_action(state_id)

            # Counts the episode as 'seen'
            self.seen += 1

        # Registers only occur if not in test mode
        if not self.test_mode:
            # Adds the state_id to the episode's history
            self.visited.append(state_id)

            # Adds the selected action to the episode's history
            self.actions.append(action)

        return action
