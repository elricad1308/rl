import pickle
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
        # Episode count for the agent
        self.episode = 0

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
        # are truncated to just 1 decimal. Two last items of the
        # observation indicate if legs had made contact, and are
        # considered meaningless for this agent
        state = []

        for i, element in enumerate(obs):
            if i < 6:
                n = int(round(element, 1) * 10)
                state.append(n)
        # t = tuple(f"{x:.1f}" for x in obs)

        # return ' '.join(t)
        return '.'.join(state)

    def debug(self, ret):
        """Print debug information."""
        e = self.episode
        r = ret
        num_states = len(self.states)
        size_episode = len(self.visited)

        perc_seen = (self.seen / size_episode) * 100
        perc_unseen = (self.unseen / size_episode) * 100

        print(
          f"Episode: {e}\t"
          f"Reward: {r:.2f}\t"
          f"Success: {self.success}\t"
          f"Ep. size: {size_episode} "
          f"({perc_seen:.2f}% seen / {perc_unseen:.2f}% unseen)\t"
          f"States: {num_states}"
        )

    def load(self, filename):
        data = pickle.load(open(filename, "rb"))

        self.policy = data[0]
        self.q = data[1]
        self.states = data[2]

        self.reset()

    def policy_evaluation(self):
        """Evaluate and update the current policy."""
        G = 0

        # Iterates for each time step
        for t, state in enumerate(self.visited):
            # Last state does not have a reward
            if t < len(self.visited) - 1:
                # Update the return for time t
                G = (self.gamma * G) + self.rewards[t + 1]

                s_t = state
                a_t = self.actions[t]
                q_n = self.q[s_t][a_t]
                n = self.n[s_t][a_t]

                # Updates the q(s, a) estimate
                self.q[s_t][a_t] = q_n + (1.0 / n) * (G - q_n)

                # Finds the action with max q(s, a)
                max_a = 0
                max_q = self.q[s_t][0]

                for a in range(LUNAR_LANDING_ACTIONS):
                    if(self.q[s_t][a] > max_q):
                        max_q = self.q[s_t][a]
                        max_a = a

                # Updates the probabilities under the current policy
                for a in range(LUNAR_LANDING_ACTIONS):
                    x = self.epsilon / LUNAR_LANDING_ACTIONS
                    if a == max_a:
                        self.policy[s_t][a] = 1.0 - self.epsilon + x
                    else:
                        self.policy[s_t][a] = x

        self.episode += 1

        return G

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

        # Number of new episodes created in current episode
        self.unseen = 0

        # List containing states VISITED in current episode
        self.visited = list()

    def save(self, filename):
        """Save the current state of the agent.

        The info saved includes the policy, the action-value estimate
        (q) and the list of known states.

        The actions, rewards and visited states are not saved, since
        they are only relevant for the current episode. The number
        of episodes, value of epsilon, gamma and number of times each
        action is selected neither are saved, since they are not
        necesary for future iterations.

        A list is created with the policy, the q estimate and the
        states (in that order), and then the list is pickled on the
        file with the given 'filename'.

        Args:
          filename (string): the name of the destiny file.

        """
        data = [
          self.policy,
          self.q,
          self.states
        ]

        pickle.dump(data, open(filename, "wb"))

    def select_action(self, state_id):
        """Apply an epsilon-soft policy to select an action.

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
            coin_toss = random.random()
            sum = self.policy[state_id][0]
            action = 0

            while sum < coin_toss:
                action += 1
                sum += self.policy[state_id][action]

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
            state_id = len(self.states)
            self.states[state] = state_id

            # A never-seen-before state obviously does not containg
            # an action on the policy, and thus an equiprobable policy
            # is created for it
            self.policy[state_id] = np.full(LUNAR_LANDING_ACTIONS, 1.0 / LUNAR_LANDING_ACTIONS)
            action = random.randrange(0, LUNAR_LANDING_ACTIONS)

            # And also it does not have an estimate of the action-state
            # function nor the returns for any of the actions.
            self.q[state_id] = np.zeros(LUNAR_LANDING_ACTIONS)

            # We also create the array to store the amount of times the
            # state and each action are selected
            self.n[state_id] = np.zeros(LUNAR_LANDING_ACTIONS)

            # Registers the state as unseen
            self.unseen += 1
        # Otherwise, recovers the state_id for that state, and gets a
        # epsilon-greedy action for it
        else:
            state_id = self.states[state]

            # Selects an action according to policy
            action = self.select_action(state_id)

            # Actions loaded from save files does not have a history
            if state_id not in self.n:
                self.n[state_id] = np.zeros(LUNAR_LANDING_ACTIONS)

            # Registers the state as seen
            self.seen += 1

        # Adds the state_id to the episode's history
        self.visited.append(state_id)

        # Adds the selected action to the episode's history
        self.actions.append(action)

        # Increase the number of times pair state-action is selected
        self.n[state_id][action] += 1

        return action

    def sim_visit_state(self, state):
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
            # self.states[state] = state_id

            # A never-seen-before state obviously does not containg
            # an action on the policy, and thus an equiprobable policy
            # is created for it
            # self.policy[state_id] = np.full(LUNAR_LANDING_ACTIONS, 1.0 / LUNAR_LANDING_ACTIONS)
            action = random.randrange(0, LUNAR_LANDING_ACTIONS)

            # And also it does not have an estimate of the action-state
            # function nor the returns for any of the actions.
            # self.q[state_id] = np.zeros(LUNAR_LANDING_ACTIONS)

            # We also create the array to store the amount of times the
            # state and each action are selected
            # self.n[state_id] = np.zeros(LUNAR_LANDING_ACTIONS)

            # Registers the state as unseen
            self.unseen += 1
        # Otherwise, recovers the state_id for that state, and gets a
        # epsilon-greedy action for it
        else:
            state_id = self.states[state]

            # Selects an action according to policy
            action = self.select_action(state_id)

            # Actions loaded from save files does not have a history
            if state_id not in self.n.keys():
                self.n[state_id] = np.zeros(LUNAR_LANDING_ACTIONS)

            # Registers the state as seen
            self.seen += 1

        # Adds the state_id to the episode's history
        self.visited.append(state_id)

        # Adds the selected action to the episode's history
        self.actions.append(action)

        # Increase the number of times pair state-action is selected
        # self.n[state_id][action] += 1

        return action
