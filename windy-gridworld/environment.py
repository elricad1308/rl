import os
import random
import time
import curses as crs
import numpy as np

# ================================================================== #
#                        Configurable options                        #
# ================================================================== #

DELAY = 0.1
"""Time (in seconds) to wait between renders"""

GOAL_X = 7
"""X coordinate of the goal position."""

GOAL_Y = 3
"""Y coordinate of the goal position."""

MAX_TIME = 1000
"""Maximum amount of time steps allowed before termination."""

START_X = 0
"""X coordinate of the start position."""

START_Y = 3
"""Y coordinate of the start position."""

WIND = np.array([
    0, 0, 0, 1, 1, 1, 2, 2, 1, 0
])
"""Array of wind strengths. It should contain WORLD_W elements."""

WORLD_H = 7
"""Height (in cells) of the gridworld"""

WORLD_W = 10
"""Width (in cells) of the gridworld"""

# ================================================================== #
#                  Class constants (DO NOT MODIFY!)                  #
# ================================================================== #

ACTION_STAND = 0
"""Action that makes the agent stand on place"""

ACTION_NORTH = 1
"""Action that moves the agent one cell up"""

ACTION_EAST = 2
"""Action that moves the agent one cell left"""

ACTION_SOUTH = 3
"""Action that moves the agent one cell down"""

ACTION_WEST = 4
"""Action that moves the agent one cell right"""

ACTION_NORTH_EAST = 5
"""Action that moves the agent one cell up and one cell left"""

ACTION_SOUTH_EAST = 6
"""Action that moves the agent one cell down and one cell left"""

ACTION_SOUTH_WEST = 7
"""Action that moves the agent one cell down and one cell right"""

ACTION_NORTH_WEST = 8
"""Action that moves the agent one cell up and one cell right"""


class Environment(object):
    """Implement a simple Windy Gridworld environment.

    A gridworld is a rectangular space formed by cells, in which the
    agent can move. On the most simple of the variants there are four
    actions: north, south, east and west, that deterministically cause
    the agent to move one cell in the respective direction on the grid.
    There is also a 'extended' set of actions which can move the agent
    to the diagonal adjacent cells of its current position.

    The gridworld contains two special cells: the 'start' and the
    'goal', and the task of the agent is to reach the goal as fast as
    possible: at each time step in which the agent is not on the goal
    a reward of -1 is received. Additionaly, if the agent would perform
    an action that would take it out of the gridworld, the action is
    ignored and the agent remains on its position (but a -1 reward is
    still given).

    The addition to a standard gridworld is the presence of 'wind',
    which varies from column to column. The wind causes the agent to
    move additional cells to the north on its next action, depending
    on the wind 'strength' (which is indicated as an integer below each
    column). The wind can also be stochastic, which means that with
    certain probability, the wind strength shown will be stronger or
    weaker.

    This class provides the standard methods provided by gym
    environments to simulate any algorithm.

    Attributes:
      - action (int): the action selected to perform on the current
          time step.

      - actions (set): the set of valid actions recognized by the
          environment.

      - done (bool): a flag that indicates if the episode is done.

      - goalX (int): the X coordinate of the goal.

      - goalY (int): the Y coordinate of the goal.

      - grid (numpy.array): a numpy array that works as the
          representation of the world.

      - posX (int): the X coordinate of the agent.

      - posY (int): the Y coordinate of the agent.

      - stdscr (window): the representation of the terminal window,
          used for rendering.

      - stochastic (bool): a flag that indicates if the wind behaves
          on a stochastic way.
    """

    def __clear(self):
        """Clear the standard output."""
        if os.name == 'nt':
            _ = os.system('cls')
        else:
            _ = os.system('clear')

    def __init__(self, extended=False, stochastic=False):
        """Create a new Environment.

        Args:
          - [extended] (bool): a flag that indicates if the world
              supports the extended actions. Defaults to False.

          - [stochastic] (bool): a flag that indicates if the wind
              should behave stochastically. Defaults to False.

        """
        # Standard gridworld only have the basic four actions
        # (plus the do-nothing action)
        self.actions = set([
            ACTION_STAND,
            ACTION_NORTH,
            ACTION_EAST,
            ACTION_SOUTH,
            ACTION_WEST
        ])

        # Extended gridworld agents cans also move diagonally
        if extended:
            self.actions = self.actions.union([
              ACTION_NORTH_EAST,
              ACTION_SOUTH_EAST,
              ACTION_SOUTH_WEST,
              ACTION_NORTH_WEST
            ])

        # Representation of the gridworld
        self.grid = np.zeros((WORLD_H, WORLD_W), np.int32)

        # Flag that indicates if stochastic wind should be used
        self.stochastic = stochastic

        # Count of current iteration
        self.iteration = 0

    def __render(self, stdscr):
        """Render the environment.

        This is the true function that draws the environment to the
        stdout, using the 'curses' python module. In order to prevent
        that the system's terminal becomes unstable, this function is
        called using the 'curses.wrapper' function on the 'render'
        method.

        Args:
          - stdscr (window): the 'curses' main window object. It is
              automatically provided when this methos is called through
              the 'curses.wrapper' method.

        """
        # Defines the color pair for current position
        crs.init_pair(1, crs.COLOR_CYAN, crs.COLOR_RED)

        # Defines the color pair for next position
        crs.init_pair(2, crs.COLOR_BLACK, crs.COLOR_YELLOW)

        # Defines the color pair for goal position
        crs.init_pair(3, crs.COLOR_WHITE, crs.COLOR_GREEN)

        # Defines the color pair for episode done
        crs.init_pair(4, crs.COLOR_GREEN, crs.COLOR_CYAN)

        # Defines the color pair for info messages
        crs.init_pair(5, crs.COLOR_CYAN, crs.COLOR_BLACK)

        # Defines the color pair for error messages
        crs.init_pair(6, crs.COLOR_RED, crs.COLOR_BLACK)

        # Defines the color pair for warning messages
        crs.init_pair(7, crs.COLOR_YELLOW, crs.COLOR_BLACK)

        # Clears the output before printing
        stdscr.erase()

        # Prints the header
        stdscr.addstr("Windy GridWorld v0.1\n")
        stdscr.addstr("=" * 80)
        stdscr.addstr("\n")

        for y in range(WORLD_H):
            for x in range(WORLD_W):
                is_goal = GOAL_X == x and GOAL_Y == y
                is_next = self.nextX == x and self.nextY == y
                visits = self.grid[y][x]
                str_visits = f"{visits:^5}"

                # The current cell is drawn with an X
                if self.posX == x and self.posY == y:
                    stdscr.addstr("[")
                    if self.done:
                        stdscr.addstr("  X  ", crs.color_pair(4))
                    else:
                        stdscr.addstr("  X  ", crs.color_pair(1))
                    stdscr.addstr("]")
                # Next action is drawn with yellow foreground
                elif is_next:
                    cell_str = "  G  " if is_goal else "     "
                    stdscr.addstr("[")
                    stdscr.addstr(cell_str, crs.color_pair(2))
                    stdscr.addstr("]")
                # Goal cell is drawn with green foreground
                elif is_goal:
                    stdscr.addstr("[")
                    stdscr.addstr("  G  ", crs.color_pair(3))
                    stdscr.addstr("]")
                # Unvisited cells are drawn empty
                elif visits == 0:
                    stdscr.addstr("[     ]")
                # Visited cells draw the number of visits
                else:
                    stdscr.addstr("[")
                    stdscr.addstr(str_visits)
                    stdscr.addstr("]")

            # Prints the newline
            stdscr.addstr("\n")

        # Draws the wind strenght for each column
        stdscr.addstr("-" * 70)
        stdscr.addstr("\n")
        for x in range(WORLD_W):
            strength = self.current_wind[x]
            stdscr.addstr(f"[{strength:^5}]")

        # Prints the messages
        stdscr.addstr("\n" * 2)
        stdscr.addstr("=" * 80)
        stdscr.addstr("\n")

        str_actions = {
            ACTION_NORTH: "NORTH",
            ACTION_EAST: "EAST",
            ACTION_SOUTH: "SOUTH",
            ACTION_WEST: "WEST",
            ACTION_NORTH_EAST: "NORTHEAST",
            ACTION_SOUTH_EAST: "SOUTHEAST",
            ACTION_SOUTH_WEST: "SOUTHWEST",
            ACTION_NORTH_WEST: "NORTHWEST",
            ACTION_STAND: "STAND"
        }
        str_act = str_actions[self.action]

        stdscr.addstr(
          f"Iteration: {self.iteration}\t"
          f"t = {self.time_count}"
          f"\tAction: {str_act}\n",
          crs.color_pair(5)
        )

        if self.inf_msg != "":
            stdscr.addstr(self.inf_msg, crs.color_pair(5))
        elif self.err_msg != "":
            stdscr.addstr(self.err_msg, crs.color_pair(6))
        elif self.warn_msg != "":
            stdscr.addstr(self.warn_msg, crs.color_pair(7))

        # Flushes to stdout
        stdscr.refresh()
        time.sleep(DELAY)

    def close(self):
        pass

    def render(self):
        """Render the environment."""
        crs.wrapper(self.__render)

    def reset(self):
        """Reset the environment to its initial configuration."""
        # Coordinates of the agent (as shown on the book)
        self.posX = START_X
        self.posY = START_Y

        # Coordinates for the next state of the agent
        # Initially, its next action is to stand idle.
        self.nextX = START_X
        self.nextY = START_Y

        # Action to perform (initially, nothing)
        self.action = ACTION_STAND

        # Array that holds the current wind speed
        self.current_wind = np.copy(WIND)

        # Flag that indicates if episode is done
        self.done = False

        # Increase the iteration number
        self.iteration += 1

        # Time count (to prevent too long episodes)
        self.time_count = 0

        # Wind push for next state
        self.wind_push = self.current_wind[START_X]

        # Messages
        self.inf_msg = ""
        self.err_msg = ""
        self.warn_msg = ""

        # Sends the initial observation
        return np.array([
            self.posX,
            self.posY,
            self.wind_push
        ])

    def step(self, action):
        """Performs a time step.

        If the episode is not done (that is, if the goal hasn't been)
        reached, the action provided is checked against the list of
        valid actions: if invalid, no action is performed.

        If the given action is valid, then the agent moves to the
        'next' position it has stored, and the wind push is applied to
        its vertical position. If the current position would cause the
        agent to leave the world, its position is adjusted accordingly.

        Then the 'next' position is computed, based on the given
        action. If the action would cause the 'next' position to fall
        of the world, its adjusted accordingly.

        As return value, the method sends a 4-element tuple: the first
        one being the 'observation', which is a 2-element numpy array
        with the 'x' and 'y' coordinates of the agent's current
        position. The second element is the reward, which as defined
        by the problem, is always '-1' per time step. The third element
        is flag that indicates whether the agent has reached its goal
        or not. Finally, the four element is an empty string, provided
        only to adhere to Gym's environment 'step' interface.

        If the episode is done (that is, if the goal has been reached),
        then no action is performed, and an empty tuple is returned.

        Args:
          - action (int): the action to perform.

        Returns:
          (tuple) a 4-element tuple, as defined on Gym's interface
        """
        # Checks if agent is on goal
        if self.done:
            return np.array([GOAL_X, GOAL_Y]), 0, True, ""

        # Increase the time count
        self.time_count += 1

        # Checks if action is valid
        if action not in self.actions:
            self.action = ACTION_STAND
            self.err_msg = f"ERROR: Invalid action '{action}'"
        else:
            self.action = action
            self.err_msg = ""

        # Creates the 'stochastic' wind
        if self.stochastic:
            aux_wind = []

            for x in range(WORLD_W):
                coin_flip = random.random()

                if coin_flip < 1.0 / 3.0:
                    aux_wind.append(WIND[x])
                elif coin_flip < 2.0 / 3.0:
                    aux_wind.append(WIND[x] + 1)
                else:
                    aux_wind.append(max(WIND[x] - 1, 0))

            self.current_wind = np.array(aux_wind)
        else:
            self.current_wind = np.copy(WIND)

        # Moves the agent to its next position
        self.posX = self.nextX
        self.posY = self.nextY

        # Applies and updates the wind push
        self.posY -= self.wind_push
        self.nextY = self.posY
        self.wind_push = self.current_wind[self.posX]

        # Keeps the agent within the grid
        self.posX = 0 if self.posX < 0 else self.posX
        self.posX = WORLD_W - 1 if self.posX >= WORLD_W else self.posX
        self.posY = 0 if self.posY < 0 else self.posY
        self.posY = WORLD_H - 1 if self.posY >= WORLD_H else self.posY

        # Checks if goal has been reached
        self.done = (self.posX == GOAL_X and self.posY == GOAL_Y) or self.time_count >= MAX_TIME

        # Applies the action to the next position of the agent
        if not self.done and self.action == ACTION_NORTH:
            self.nextY -= 1

        if not self.done and self.action == ACTION_NORTH_EAST:
            self.nextX += 1
            self.nextY -= 1

        if not self.done and self.action == ACTION_EAST:
            self.nextX += 1

        if not self.done and self.action == ACTION_SOUTH_EAST:
            self.nextX += 1
            self.nextY += 1

        if not self.done and self.action == ACTION_SOUTH:
            self.nextY += 1

        if not self.done and self.action == ACTION_SOUTH_WEST:
            self.nextX -= 1
            self.nextY += 1

        if not self.done and self.action == ACTION_WEST:
            self.nextX -= 1

        if not self.done and self.action == ACTION_NORTH_WEST:
            self.nextX -= 1
            self.nextY -= 1

        # Checks if agent would leave the grid to show a warning
        if not self.done and self.nextX < 0 or self.nextX >= WORLD_W:
            self.warn_msg = "WARNING! Agent fell of the world"
        elif not self.done and self.nextY < 0 or self.nextY >= WORLD_H:
            self.warn_msg = "WARNING! Agent fell of the world"
        else:
            self.warn_msg = ""

        # Keeps the next action within the grid
        self.nextX = 0 if self.nextX < 0 else self.nextX
        self.nextX = WORLD_W - 1 if self.nextX >= WORLD_W else self.nextX
        self.nextY = 0 if self.nextY < 0 else self.nextY
        self.nextY = WORLD_H - 1 if self.nextY >= WORLD_H else self.nextY

        # Checks if agent is on goal
        if self.done:
            self.inf_msg = "Success! Goal reached."
        else:
            self.inf_msg = ""

        # Observation is a numpy array with current X and Y,
        # and next wind push
        observation = np.array([
            self.posX,
            self.posY,
            self.wind_push
        ])

        # Reward is always -1 per time step
        reward = -1

        # Episode is done if goal is reached
        done = self.done

        # Return a tuple as defined on Gym's 'step' interface
        return observation, reward, done, ""
