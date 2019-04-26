import gym
import offpmc

def episode():
    obs = env.reset()

    for i in range(15):
        obs, reward, done, info = env.step(0)
        state = algo.create_state(obs)
        print(f"Observation: {str(obs)}\nState: {str(state)}\n\n")


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    algo = offpmc.OffPolicyMonteCarlo()

    episode()
