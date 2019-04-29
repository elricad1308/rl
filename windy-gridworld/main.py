import sys
import getopt
import environment

import sarsa
import q_learning
import expected_sarsa as expected


def episode():
    """Perform a complete episode.

    First, the environment and the algorithm are reset to their default
    states. Also, the initial observation is obtained from the
    environment.

    Then the initial observation is used to select the initial action,
    and then the episode loop starts: at each time step the environment
    is rendered (if desired), the observation and rewards are obtained
    from the environment and send to the algorithm to process them.

    Finally, information about the state of the agent is printed.

    """
    # Resets the environment
    obs = env.reset()
    action = algo.reset(obs)

    # This flag determines the end of the episode
    done = False

    # Repeat until the end of the episode
    while not done:
        # Only renders the environment if told so
        if render:
            env.render()

        # Gets the observation and reward from environment
        obs, reward, done, info = env.step(action)

        # Send observation and rewards to the algorithm
        # (only if the simulation is not done)
        if not done:
            action = algo.step(obs, reward)

    if not render:
        algo.debug()


def print_usage(small=True):
    """Print help about the usage of the module.

    Args:
      - [small] (bool): a flag that indicates if the short version of
          the usage should be printed. Defaults to True.

    """
    if small:
        usage = (
          f"USAGE: main.py "
          f"-m <sarsa | qlearning | expected> "
          f"[-a|--alpha <alpha>] "
          f"[-e|--epsilon <epsilon>] "
          f"[-g|--gamma <gamma>] "
          f"[-i|--input <inputfile>] "
          f"[-o|--output <outputfile>] "
          f"[-c|--cycles <cycles>] "
          f"[-h|--help] "
          f"[-r|--render] "
          f"[-k|--king] "
          f"[-s|--stochastic]"
        )
    else:
        usage = (
          f"main.py | Executes the Lunar Landing environment.\n"
          f"  Arguments:\n"
          f"\t-m|--method <sarsa | qlearning | expected>\n"
          f"\t[-a | --alpha]\t\t: Alpha value for the algorithm.\n"
          f"\t[-c | --cycles <cycles>]\t\t: Number of episodes to run.\n"
          f"\t[-e | --epsilon]\t\t: Epsilon value for the algorithm.\n"
          f"\t[-g | --gamma]\t\t: Gamma value for the algorithm.\n"
          f"\t[-h | --help]\t\t: Prints this message.\n"
          f"\t[-i | --input <inputfile>]\t: Load agent from <inputfile>.\n"
          f"\t[-k | --king]\t\t: Allow the agent to use full king's moves.\n"
          f"\t[-o | --output <outputfile>]\t: Save agent to <outputfile>.\n"
          f"\t[-r | --render]\t\t: Render the animation during execution.\n"
          f"\t[-s | --stochastic]\t: Run the stochastic wind simulation.\n"
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
      "king",
      "method=",
      "output=",
      "render",
      "stochastic"
    ]

    options = "a:c:e:g:hi:km:o:rs"

    try:
        opts, args = getopt.getopt(sys.argv[1:], options, long_options)
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    # Alpha, cycles, epsilon and gamma values
    alpha = 0.9
    cycles = 1000
    epsilon = 0.05
    gamma = 0.9

    # Agent's input file
    input = ""

    # Flag for king's moves
    king = False

    # Method used for solving
    method = "sarsa"

    # Agent's output file
    output = ""

    # Flag to render the simulation
    render = False

    # Flag for stochastic wind
    stochastic = False

    # Parse the received arguments
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
        elif opt in ("-h", "--help"):
            print_usage(False)
            sys.exit()

        # Option to load an agent
        elif opt in ("-i", "--input"):
            input = arg

        # Option to run with full king's moves
        elif opt in ("-k", "--king"):
            king = True

        # Option to set the method
        elif opt in ("-m", "--method"):
            method = arg

        # Option to save the agent
        elif opt in ("-o", "--output"):
            output = arg

        # Option to render the simulation
        elif opt in ("-r", "--render"):
            render = True

        # Option to use stochastic wind
        elif opt in ("-s", "--stochastic"):
            stochastic = True

    # Create an instance of the algorithm
    if method == "sarsa":
        algo = sarsa.Algorithm(alpha, epsilon, gamma, king, stochastic)
    elif method == "qlearning":
        algo = q_learning.Algorithm(alpha, epsilon, gamma, king, stochastic)
    elif method == "expected":
        algo = expected.Algorithm(alpha, epsilon, gamma, king, stochastic)
    else:
        print_usage()
        sys.exit(2)

    # Loads the environment
    env = environment.Environment(king, stochastic)

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
