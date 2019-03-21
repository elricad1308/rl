import random
import numpy as np

LUNAR_LANDING_ACTIONS = 4

class OnPolicyMonteCarlo(object):

    def __init__(self, epsilon, gamma):
        # Epsilon attribute for the epsilon-greedy policy.
        self.epsilon = epsilon

        # Gamma attribute for the return formula.
        self.gamma = gamma

        # Dictionary containing ALL seen states. It functions as a one
        # to one mapping from observed 'states' to an assigned ID, in
        # order to make structures more efficient.
        self.states = dict()

        # Dictionary containing the policy. It contains as keys all
        # seen states IDs, and as values de greedy action for that
        # state
        self.policy = dict()

        # Dictionary containing the estimate of the action-state value
        # function.
        self.q = dict()

        # List containing states VISITED in current episode
        self.visited = list()

    def create_state(self, obs):
        # To reduce the number of states, values from the observation
        # are truncated to just 1 decimal
        t = tuple(f"{x:.1f}" for x in obs)

        return ' '.join(t)

    def visit_state(self, state):
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
            self.returns[state_id] = np.zeros(LUNAR_LANDING_ACTIONS)
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

        # Adds the state_id to the episode's history ONLY if it has not
        # been added before (first-visit MC)
        if state not in self.visited:
            self.visited.append(state_id)

        return action



    def step(self, obs):
        state = self.create_state(obs)

        self.visit_state(state)
