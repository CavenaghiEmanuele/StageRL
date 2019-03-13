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
    lista2=[]

    env = gym.make("FrozenLake8x8-v0")

    agent_info = MCA.monte_carlo_e_soft(env, episodes=100, epsilon=0.01)


    for _ in range(500): #play for 100 games

        agent_info = MCA.monte_carlo_e_soft(env, episodes=1000, policy=agent_info["policy"],
            state_action_table=agent_info["state_action_table"], returns=agent_info["returns"], epsilon=0.01)

        lista.append(MCA.test_policy(agent_info["policy"], env))


    plt.plot(lista)



    agent_info2 = MCA.monte_carlo_e_soft(env, episodes=100, epsilon=0.001)

    for _ in range(500):

        agent_info2 = MCA.monte_carlo_e_soft(env, episodes=1000, policy=agent_info2["policy"],
            state_action_table=agent_info2["state_action_table"], returns=agent_info2["returns"], epsilon=0.001)

        lista2.append(MCA.test_policy(agent_info2["policy"], env))


    plt.plot(lista2)



    agent_info3 = MCA.monte_carlo_e_soft(env, episodes=100, epsilon=0.05)

    for _ in range(500):

        agent_info3 = MCA.monte_carlo_e_soft(env, episodes=1000, policy=agent_info3["policy"],
            state_action_table=agent_info3["state_action_table"], returns=agent_info3["returns"], epsilon=0.05)

        lista3.append(MCA.test_policy(agent_info3["policy"], env))


    plt.plot(lista3)


    plt.ylabel('% wins')
    plt.xlabel('Number of games')
    plt.legend(["epsilon=0.01", "epsilon=0.001", "epsilon=0.05"], loc='upper left')
    plt.show()






#    print(list)
