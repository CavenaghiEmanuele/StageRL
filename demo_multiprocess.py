import gym
import numpy as np
import matplotlib.pyplot as plt
import operator
from IPython.display import clear_output
from time import sleep
import itertools
from multiprocessing import Pool
import os


import agents.monte_carlo_agent as MCA
import agents.dynamic_programming_agent as DPA


def input_for_agent(n_agent):

    print()
    print("*************************************************")
    print("*                    AGENT " + str(n_agent+1) + "                    *")
    print("*************************************************")

    agent_type = input("Insert the agent type: ")
    agent = {}


    if agent_type == "MonteCarlo" or agent_type == "MC":
        n_games = int(input("Insert the number of games: "))
        n_episodes = int(input("Insert the number of episodes for each game: "))
        epsilon = float(input("Insert the parameter epsilon: "))

        agent ={
            "type": agent_type,
            "n_games": n_games,
            "n_episodes": n_episodes,
            "epsilon": epsilon
        }

    elif agent_type == "Dynamic programming" or agent_type == "DP":
        gamma = float(input("Insert the parameter gamma: "))
        theta = float(input("Insert the parameter theta: "))

        agent ={
            "type": agent_type,
            "gamma": gamma,
            "theta": theta,
        }

    return agent


def create_custom_enviroment():

        gym.register(
            id='FrozenLakeNotSlippery8x8-v0',
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={'map_name' : '8x8', 'is_slippery': False},
            max_episode_steps=1000,
        )

        gym.register(
            id='FrozenLakeNotSlippery4x4-v0',
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={'map_name' : '4x4', 'is_slippery': False},
            max_episode_steps=1000,
        )


def create_legend_string(agent):

    string = ""

    if agent["type"] == "MonteCarlo" or agent["type"] == "MC":
        return "MonteCarlo, epsilon=" + str(agent["epsilon"]) + ", n_games=" + str(agent["n_games"]) + ", n_episodes=" + str(agent["n_episodes"])

    elif agent["type"] == "Dynamic programming" or agent["type"] == "DP":
        return "Dynamic programming, gamma=" + str(agent["gamma"]) + ", theta=" + str(agent["theta"])


def run_agent(agent_dict):
    if agent_dict["type"] == "MonteCarlo" or agent_dict["type"] == "MC":

        dict_result = MCA.run_agent(
            enviroment,
            agent_dict["n_games"],
            agent_dict["n_episodes"],
            epsilon = agent_dict["epsilon"]
        )

    elif agent_dict["type"] == "Dynamic programming" or agent_dict["type"] == "DP":

        dict_result = DPA.run_agent(
            enviroment,
            agent_dict["gamma"],
            agent_dict["theta"],
        )

    test_result = dict_result["tests_result"]

    return test_result


if __name__ == '__main__':

    agents_list = []
    tests_result = []
    create_custom_enviroment()


    enviroment_name = input("Insert the enviroment name: ")
    n_agents = int(input("Insert the number of agents: "))

    for i in range(n_agents):
        agents_list.append(input_for_agent(i))


    enviroment = gym.make(enviroment_name) #Creazione ambiente


    pool = Pool(len(os.sched_getaffinity(0))) #creo un pool di processi
    results = pool.starmap(run_agent, zip(agents_list)) #Ogni agente viene affidato ad un processo

    pool.close()
    pool.join() # attendo che tutti gli agenti abbiano terminato il trining per poi prseguire


    #per ogni agente recupero il risultato dei test
    for agent in range(len(agents_list)):
        #Aggiunta della lista dei risultati al grafico
        plt.plot(results[agent])


    legend = []
    for agent in agents_list:
        legend.append(create_legend_string(agent))

    plt.legend(legend, loc='upper left')
    plt.show()
