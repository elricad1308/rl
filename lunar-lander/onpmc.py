import random
import numpy as np

LUNAR_LANDING_ACTIONS = 4


class OnPolicyMonteCarlo(object):
    """Implement an On-Policy Monte Carlo agent for Lunar Landing.

    Attributes:
      - actions (list): a list that contains the actions selected on
          the current episode.

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
      - states (dict): the dictionary containing ALL known states. It
          maps from states as received from the observation (in the
          case of Lunar Landing, an 8-element array) to a natural int
          used as the State ID.

      - visited (list): a list that contains the IDs of the visited
          states on the current episode.

    """

    def __init__(self, epsilon, gamma):
        """Create a new agent.

        Args:
          - epsilon (float): the value to use for the epsilon-greedy
              policy selection.

          - gamma (float): the value to use as the discount rate for
              the return.

        """
        # Epsilon attribute for the epsilon-greedy policy.
        self.epsilon = epsilon

        # Gamma attribute for the return formula.
        self.gamma = gamma

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

        # Prepares the agent for a new episode.
        self.reset()

    def create_state(self, obs):
        """Transform an observation to a state.

        The Lunar Landing provides at each time step an 8 element numpy
        array, containing information such as the X and Y coordinates
        of the pad (which are continuous instead of discrete), the
        speed, angle and angular speed of the pad, and flags that
        indicate whether the pad has made contact with the moon or not.

        Due to all values being real, the amount of possible states is
        overwhelming, and thus, in order to maintain efficiency, the
        values from the observation array are truncated to just one
        decimal position, and then join together as a single string,
        which is the representation that the agent uses to identify
        states.

        Args:
          - obs (numpy.array) the observation array, as sent from the
              environment.

        Returns:
          (string) a 'hash' string of the state, created by truncating
            the values and joining them together.

        """
        # To reduce the number of states, values from the observation
        # are truncated to just 1 decimal
        t = tuple(f"{x:.1f}" for x in obs)

        return ' '.join(t)

    def debug(self):
        """Print debug information."""
        num_states = len(self.states)

        print(f"Agent known {num_states} states")

    def reset(self):
        """Prepare the agent for a new episode.

        This method re-creates all the data structures that change with
        each episode (actions, rewards and visited states).

        """
        # List containing actions selected in current episode
        self.actions = list()

        # List containing all rewards received on current episode.
        self.rewards = list()

        # List containing states VISITED in current episode
        self.visited = list()

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
        if state not in self.states.keys():
            state_id = len(self.states)
            self.states[state] = state_id

            # A never-seen-before state obviously does not containg
            # an action on the policy
            action = random.randrange(0, LUNAR_LANDING_ACTIONS)
            self.policy[state_id] = action

            # And also it does not have an estimate of the action-state
            # function nor the returns for any of the actions.
            self.q[state_id] = np.zeros(LUNAR_LANDING_ACTIONS)
            # self.returns[state_id] = np.zeros(LUNAR_LANDING_ACTIONS)
        # Otherwise, recovers the state_id for that state, and gets a
        # epsilon-greedy action for it
        else:
            state_id = self.states[state]

            # With probability 1 - epsilon, greedy action is chosed
            if random.random() < (1.0 - self.epsilon):
                action = self.policy[state_id]
            # With probability epsilon, a random action is chosed
            else:
                action = random.randrange(0, LUNAR_LANDING_ACTIONS)

        # Adds the state_id to the episode's history
        self.visited.append(state_id)

        # Adds the selected action to the episode's history
        self.actions.append(action)

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

        action = self.visit_state(state)

        # A reward of None indicates that this is the first time step.
        if reward is not None:
            self.rewards.append(reward)

        return action
