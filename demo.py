import gym
import numpy as np
import operator
from IPython.display import clear_output
from time import sleep
import itertools
import tqdm
import agents.MonteCarloAgent as MCA

tqdm.monitor_interval = 0



if __name__ == '__main__':
    env = gym.make("FrozenLake8x8-v0")
    policy = MCA.monte_carlo_e_soft(env, episodes=1000)
    print(MCA.test_policy(policy, env))
