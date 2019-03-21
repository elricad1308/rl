import sys
sys.path.append("/usr/local/lib/python3.7/site-packages")
import gym
import onpmc

env = gym.make('LunarLander-v2')
observation = env.reset()



algo = onpmc.OnPolicyMonteCarlo(env)

def ff(action):
    for i in range(24):
        obs, reward, done, info = env.step(action)
        env.render()
        x = obs[0]
        y = obs[1]
        state = algo.create_state(obs)
        # print(f"Coordinates: ({x:.2f}, {y:.2f})\tReward: {reward:.2f}")
        print(f"State: {state}")
