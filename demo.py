import gym
import numpy as np
import matplotlib.pyplot as plt
import operator
from IPython.display import clear_output
from time import sleep
import itertools
from argparse import ArgumentParser as parser

import agents.monte_carlo_agent as MCA
import agents.dynamic_programming_agent as DPA
import agents.q_learning as QLA


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

    elif agent_type == "Q learning" or agent_type == "QL":
        alpha = float(input("Insert the parameter alpha (learning rate): "))
        gamma = float(input("Insert the parameter gamma: "))
        epsilon = float(input("Insert the parameter epsilon: "))
        n_games = int(input("Insert the number of games: "))
        n_episodes = int(input("Insert the number of episodes for each game: "))

        agent ={
            "type": agent_type,
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon,
            "n_games": n_games,
            "n_episodes": n_episodes,
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

    elif agent["type"] == "Q learning" or agent["type"] == "QL":
        return "Q learning, alpha=" + str(agent["alpha"]) + ", gamma=" + str(agent["gamma"]) + ", epsilon=" + str(agent["epsilon"]) + ", n_games=" + str(agent["n_games"]) + ", n_episodes=" + str(agent["n_episodes"])



if __name__ == '__main__':

    agents_list = []
    create_custom_enviroment()

    enviroment_name = input("Insert the enviroment name: ")
    n_agents = int(input("Insert the number of agents: "))

    for i in range(n_agents):
        agents_list.append(input_for_agent(i))


    enviroment = gym.make(enviroment_name) #Creazione ambiente

    for agent in agents_list:

        if agent["type"] == "MonteCarlo" or agent["type"] == "MC":

            dict_result = MCA.run_agent(
                enviroment,
                agent["n_games"],
                agent["n_episodes"],
                epsilon = agent["epsilon"]
            )

        elif agent["type"] == "Dynamic programming" or agent["type"] == "DP":

            dict_result = DPA.run_agent(
                enviroment,
                gamma = agent_dict["gamma"],
                theta = agent_dict["theta"],
            )

        elif agent["type"] == "Q learning" or agent["type"] == "QL":

            dict_result = QLA.run_agent(
                enviroment,
                alpha = agent_dict["alpha"],
                gamma = agent_dict["gamma"],
                epsilon = agent_dict["epsilon"],
                n_games = agent_dict["n_games"],
                n_episodes = agent_dict["n_episodes"]
            )
            
        plt.plot(dict_result["tests_result"])


    legend = []
    for agent in agents_list:
        legend.append(create_legend_string(agent))

    plt.legend(legend, loc='upper left')
    plt.show()
