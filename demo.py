import gym
import numpy as np
import matplotlib.pyplot as plt
import operator
from IPython.display import clear_output
from time import sleep
import itertools
import tqdm
import agents.MonteCarloAgent as MCA

from pprint import pprint


tqdm.monitor_interval = 0



if __name__ == '__main__':

    lista=[]
    env = gym.make("FrozenLake8x8-v0")
    policy = MCA.monte_carlo_e_soft(env, episodes=10000)

    for _ in range(100):

        lista.append(MCA.test_policy(policy, env))
        policy = MCA.monte_carlo_e_soft(env, episodes=100, policy=policy)


#    print(list)

    plt.plot(lista)
    plt.ylabel('some numbers')
    plt.show()
