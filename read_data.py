import matplotlib.pyplot as plt
from pprint import pprint
import json


def input_for_agent():


    agent_type = input("Insert the agent type: ")
    agent = {}

    print()
    print("*************************************************")
    print("*                    AGENT                      *")
    print("*************************************************")

    if agent_type == "MonteCarlo" or agent_type == "MC":
        n_games = int(input("Insert the number of games: "))
        n_episodes = int(input("Insert the number of episodes for each game: "))
        epsilon = float(input("Insert the parameter epsilon: "))
        gamma = float(input("Insert the parameter gamma: "))

        agent ={
            "type": "MonteCarlo",
            "n_games": n_games,
            "n_episodes": n_episodes,
            "epsilon": epsilon,
            "gamma": gamma
        }

    elif agent_type == "Dynamic programming" or agent_type == "DP":
        gamma = float(input("Insert the parameter gamma: "))
        theta = float(input("Insert the parameter theta: "))

        agent ={
            "type": "Dynamic programming",
            "gamma": gamma,
            "theta": theta
        }

    elif agent_type == "Q learning" or agent_type == "QL":
        alpha = float(input("Insert the parameter alpha (learning rate): "))
        gamma = float(input("Insert the parameter gamma: "))
        epsilon = float(input("Insert the parameter epsilon: "))
        n_games = int(input("Insert the number of games: "))
        n_episodes = int(input("Insert the number of episodes for each game: "))

        agent ={
            "type": "Q learning",
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon,
            "n_games": n_games,
            "n_episodes": n_episodes
        }

    elif agent_type == "n-step SARSA" or agent_type == "NSS":
        n_step = int(input("Insert the number of step for the agent: "))
        alpha = float(input("Insert the parameter alpha (learning rate): "))
        gamma = float(input("Insert the parameter gamma: "))
        epsilon = float(input("Insert the parameter epsilon: "))
        n_games = int(input("Insert the number of games: "))
        n_episodes = int(input("Insert the number of episodes for each game: "))

        agent ={
            "type": "n-step SARSA",
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon,
            "n_games": n_games,
            "n_episodes": n_episodes,
            "n_step": n_step
        }

    return agent


def create_agent_params_string(agent):

    string = ""

    if agent["type"] == "MonteCarlo" or agent["type"] == "MC":
        return "Epsilon= " + str(agent["epsilon"]) + ", gamma= " + str(agent["gamma"]) + ", n_games= " + str(agent["n_games"]) + ", n_episodes= " + str(agent["n_episodes"])

    elif agent["type"] == "Dynamic programming" or agent["type"] == "DP":
        return "Gamma= " + str(agent["gamma"]) + ", theta= " + str(agent["theta"])

    elif agent["type"] == "Q learning" or agent["type"] == "QL":
        return "Alpha= " + str(agent["alpha"]) + ", gamma= " + str(agent["gamma"]) + ", epsilon= " + str(agent["epsilon"]) + ", n_games= " + str(agent["n_games"]) + ", n_episodes= " + str(agent["n_episodes"])

    elif agent["type"] == "n-step SARSA" or agent["type"] == "NSS":
        return "N-step= " + str(agent["n_step"]) + ", alpha= " + str(agent["alpha"]) + ", gamma= " + str(agent["gamma"]) + ", epsilon= " + str(agent["epsilon"]) + ", n_games= " + str(agent["n_games"]) + ", n_episodes= " + str(agent["n_episodes"])


def create_legend_string(agent):

    string = ""

    if agent["type"] == "MonteCarlo" or agent["type"] == "MC":
        return "MonteCarlo, epsilon=" + str(agent["epsilon"]) + ", gamma=" + str(agent["gamma"]) + ", n_games=" + str(agent["n_games"]) + ", n_episodes=" + str(agent["n_episodes"])

    elif agent["type"] == "Dynamic programming" or agent["type"] == "DP":
        return "Dynamic programming, gamma=" + str(agent["gamma"]) + ", theta=" + str(agent["theta"])

    elif agent["type"] == "Q learning" or agent["type"] == "QL":
        return "Q learning, alpha=" + str(agent["alpha"]) + ", gamma=" + str(agent["gamma"]) + ", epsilon=" + str(agent["epsilon"]) + ", n_games=" + str(agent["n_games"]) + ", n_episodes=" + str(agent["n_episodes"])

    elif agent["type"] == "n-step SARSA" or agent["type"] == "NSS":
        return "n-step SARSA, n-step=" + str(agent["n_step"]) + ",alpha=" + str(agent["alpha"]) + ", gamma=" + str(agent["gamma"]) + ", epsilon=" + str(agent["epsilon"]) + ", n_games=" + str(agent["n_games"]) + ", n_episodes=" + str(agent["n_episodes"])



if __name__ == '__main__':


    enviroment_name = input("Insert the enviroment name: ")
    tests_moment = input("Select the test type (final, on_run, ten_perc): ")
    agent = input_for_agent()

    path = "docs/" + enviroment_name + "/" + tests_moment + "/" + agent["type"] + "/" + create_agent_params_string(agent)


    with open(path) as inputfile:
        tests_list = json.load(inputfile)


    legend = []

    for test_type in tests_list[0]:
        plt.figure(test_type)

        for test_agent in range(len(tests_list)):

            legend.append(create_legend_string(agent))
            plt.plot(tests_list[test_agent][test_type])

        plt.legend(legend, loc='upper left')
        plt.ylabel(test_type)

    plt.show()
