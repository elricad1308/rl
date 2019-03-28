import sys
sys.path.append("/usr/local/lib/python3.7/site-packages")
# sys.path.append("/lustre/users/josea/.local/lib/python3.6/site-packages")
import gym
import onpmc

def episode():
    # Resets the environtment
    obs = env.reset()
    algo.reset()

    # Choose initial action
    action = algo.step(obs)

    # This flag determines the end of the episode
    done = False

    while not done:
        env.render()
        obs, reward, done, info = env.step(action)
        action = algo.step(obs, reward)

    # final_return = algo.policy_evaluation()

    algo.debug(reward)


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    algo = onpmc.OnPolicyMonteCarlo(0.05, 0.75)

    algo.load("run150000.v1.bin")

    for i in range(1000):
        episode()

        # if i % 1000 == 0:
        #    algo.save("checkpoint.bin")

    # algo.save("run10000.v2.bin")
    env.close()
