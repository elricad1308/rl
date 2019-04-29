import sys
import getopt
import onpmc        # On-policy Monte Carlo
import offpmc       # Off-policy Monte Carlo
import sarsa
import q_learning
import expected_sarsa

# Uncomment (or add) the path to gym on your system if it cannot be
# found by the interpreter!

# Path to gym on Fedora Linux
sys.path.append("/usr/local/lib/python3.7/site-packages")

# Path to gym on the cluster
# sys.path.append("/lustre/users/josea/.local/lib/python3.6/site-packages")

import gym


def episode():
    """Perform a complete episode.

    First, the environment and the algorithm are reset to their default
    states. Also, the initial observation is obtained from the
    environment.

    Then the initial observation is used to select the initial action,
    and then the episode loop starts: at each time step the environment
    is rendered (if desired), the observation and rewards are obtained
    from the environment and send to the algorithm to process them.

    When the environment states that the episode is over, then the
    algorithm proceeds to apply the policy iteration.

    Finally, information about the state of the agent is printed.

    """
    # Resets the environtment
    obs = env.reset()

    # Choose action from initial observation
    action = algo.reset(obs)

    # This flag determines the end of the episode
    done = False

    # Repeat until the episode is done
    while not done:
        # Only renders the environment if told so
        if render:
            env.render()

        # Gets the observation and rewards from environment
        obs, reward, done, info = env.step(action)

        # Send observation and rewards to the algorithm
        # (only if the simulation is not done)
        if not done:
            action = algo.step(obs, reward)

    # Proceeds to improve the policy
    algo.policy_iteration(reward)

    # Prints information about the episode
    algo.debug()


def print_usage(small=True):
    """Print help about the usage of the module."""
    if small:
        usage = (
          f"USAGE: main.py "
          f"-m|--method <onpmc | offpmc | sarsa>"
          f"[-a|--alpha <alpha>] "
          f"[-e|--epsilon <epsilon>] "
          f"[-g|--gamma <gamma>] "
          f"[-i|--input <inputfile>] "
          f"[-o|--output <outputfile>] "
          f"[-c|--cycles <cycles>] "
          f"[-h|--help] "
          f"[-r|--render] "
          f"[-t|--test] "
        )
    else:
        usage = (
          f"main.py | Executes the Lunar Landing environment.\n"
          f"  Arguments:\n"
          f"\t-m | --method <onpmc | offpmc | sarsa | qlearning | expected>\n"
          f"\t[-a | --alpha]\t\t: Alpha value for the algorithm."
          f"\t[-e | --epsilon]\t\t: Epsilon value for the algorithm.\n"
          f"\t[-g | --gamma]\t\t: Gamma value for the algorithm.\n"
          f"\t[-i | --input <inputfile>]\t: Load agent from <inputfile>.\n"
          f"\t[-o | --output <outputfile>]\t: Save agent to <outputfile>.\n"
          f"\t[-c | --cycles <cycles>]\t\t: Number of episodes to run the simulation.\n"
          f"\t[-h | --help]\t\t: Prints this message.\n"
          f"\t[-r | --render]\t\t: Render the animation during execution.\n"
          f"\t[-t | --test]\t\t: Run the simulation on test mode.\n"
        )

    print(usage)


if __name__ == "__main__":
    long_options = [
      "alpha=",
      "cycles=",
      "epsilon=",
      "gamma=",
      "help",
      "input=",
      "method="
      "output=",
      "render",
      "test"
    ]

    options = "a:c:e:g:hi:m:o:rt"

    try:
        opts, args = getopt.getopt(sys.argv[1:], options, long_options)
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    # Alpha, epsilon, gamma and cycle values
    alpha = 0.5
    epsilon = 0.05
    gamma = 0.9
    cycles = 1000

    # Flag to load an agent
    input = ""

    # Method to use
    method = "onpmc"

    # Flag to save the agent
    output = ""

    # Flag to render the simulation
    render = False

    # Flag to run simulation on test mode
    test = False

    for opt, arg in opts:
        # Option to specify alpha
        if opt in ("-a", "--alpha"):
            alpha = float(arg)

        # Option to specify number of cycles
        elif opt in ("-c", "--cycles"):
            cycles = int(arg)

        # Option to specify epsilon
        elif opt in ("-e", "--epsilon"):
            epsilon = float(arg)

        # Option to specify gamma
        elif opt in ("-g", "--gamma"):
            gamma = float(arg)

        # Option to print help
        if opt in ("-h", "--help"):
            print_usage(False)
            sys.exit()

        # Option to load an agent
        elif opt in ("-i", "--input"):
            input = arg

        # Option to specify the agent
        elif opt in ("-m", "--method"):
            method = arg

        # Option to save the agent
        elif opt in ("-o", "--output"):
            output = arg

        # Option to render the simulation
        elif opt in ("-r", "--render"):
            render = True

        # Option to run in test mode
        elif opt in ("-t", "--test"):
            test = True

    # Create the agent to use
    if method == "onpmc":
        algo = onpmc.Algorithm(epsilon, gamma, test)
    elif method == "offpmc":
        algo = offpmc.Algorithm(epsilon, gamma, test)
    elif method == "sarsa":
        algo = sarsa.Algorithm(alpha, epsilon, gamma, test)
    elif method == "qlearning":
        algo = q_learning.Algorithm(alpha, epsilon, gamma, test)
    elif method == "expected":
        algo = expected_sarsa.Algorithm(alpha, epsilon, gamma, test)
    else:
        print_usage()
        sys.exit(2)

    # Loads the environment
    env = gym.make('LunarLander-v2')

    # If told so, load a saved agent
    if input:
        algo.load(input)

    # Run the simulation for the given number of episodes
    for _ in range(cycles):
        episode()

    # If told so, save the agent
    if output:
        algo.save(output)

    # Closes the environment before closing
    env.close()
