import itertools
from multiprocessing import Pool
import os
import errno
import json
from argparse import ArgumentParser as parser
import gym

import agents.monte_carlo as MC
import agents.dynamic_programming as DP
import agents.q_learning as QL
import agents.n_step_sarsa as NSS
import agents.n_step_sarsa_approximate as NSSA
import agents.sarsa_lambda as SL




def input_for_agent(n_agent, tests_moment):

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
        gamma = float(input("Insert the parameter gamma: "))

        agent = {
            "type": "MonteCarlo",
            "n_games": n_games,
            "n_episodes": n_episodes,
            "epsilon": epsilon,
            "gamma": gamma
        }

    elif (agent_type == "Dynamic programming" or agent_type == "DP") and tests_moment == "final":
        gamma = float(input("Insert the parameter gamma: "))
        theta = float(input("Insert the parameter theta: "))

        agent = {
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

        agent = {
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

        agent = {
            "type": "n-step SARSA",
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon,
            "n_games": n_games,
            "n_episodes": n_episodes,
            "n_step": n_step
        }

    elif agent_type == "n-step SARSA approximate" or agent_type == "NSSA":
        n_step = int(input("Insert the number of step for the agent: "))
        alpha = float(input("Insert the parameter alpha (learning rate): "))
        gamma = float(input("Insert the parameter gamma: "))
        epsilon = float(input("Insert the parameter epsilon: "))
        n_games = int(input("Insert the number of games: "))
        n_episodes = int(input("Insert the number of episodes for each game: "))

        agent = {
            "type": "n-step SARSA approximate",
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon,
            "n_games": n_games,
            "n_episodes": n_episodes,
            "n_step": n_step
        }

    elif agent_type == "SARSA lambda" or agent_type == "SL":
        n_step = int(input("Insert the number of step for the agent: "))
        alpha = float(input("Insert the parameter alpha (learning rate): "))
        gamma = float(input("Insert the parameter gamma: "))
        epsilon = float(input("Insert the parameter epsilon: "))
        lambd = float(input("Insert the parameter lambda: "))
        n_games = int(input("Insert the number of games: "))
        n_episodes = int(input("Insert the number of episodes for each game: "))

        agent = {
            "type": "SARSA lambda",
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon,
            "lambd": lambd,
            "n_games": n_games,
            "n_episodes": n_episodes,
            "n_step": n_step
        }


    if (agent_type == "Dynamic programming" or agent_type == "DP") and tests_moment != "final":
        print("Dynamic Programming agent can't have on_run or ten_perc test")
        raise

    return agent


def automatic_agent_generation(agent_type, n_games, n_episodes):

    agents_list = []

    if agent_type == "MonteCarlo" or agent_type == "MC":

        epsilon = [0.3, 0.1, 0.05, 0.01, 0.005]
        gamma = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3]

        combination = list(itertools.product(epsilon, gamma))

        for item in combination:
            agent = {
                "type": "MonteCarlo",
                "n_games": n_games,
                "n_episodes": n_episodes,
                "epsilon": item[0],
                "gamma": item[1]
            }
            for _ in range(10):
                agents_list.append(agent)

    elif (agent_type == "Dynamic programming" or agent_type == "DP"):

        gamma = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3]
        theta = [1e-3, 1e-4, 1e-5]

        combination = list(itertools.product(gamma, theta))

        for item in combination:

            agent = {
                "type": "Dynamic programming",
                "gamma": item[0],
                "theta": item[1]
            }
            for _ in range(10):
                agents_list.append(agent)


    elif agent_type == "Q learning" or agent_type == "QL":
        alpha = [0.1, 0.2, 0.3, 0.4, 0.5]
        gamma = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3]
        epsilon = [0.01, 0.05, 0.1, 0.2, 0.3]

        combination = list(itertools.product(alpha, gamma, epsilon))

        for item in combination:
            agent = {
                "type": "Q learning",
                "alpha": item[0],
                "gamma": item[1],
                "epsilon": item[2],
                "n_games": n_games,
                "n_episodes": n_episodes
            }
            for _ in range(10):
                agents_list.append(agent)

    elif agent_type == "n-step SARSA" or agent_type == "NSS":
        n_step = [2, 4, 8, 12, 16, 20]
        alpha = [0.1, 0.2, 0.3, 0.4]
        gamma = [1.0, 0.9, 0.8, 0.7]
        epsilon = [0.01, 0.05, 0.1, 0.2]

        combination = list(itertools.product(n_step, alpha, gamma, epsilon))

        for item in combination:
            agent = {
                "type": "n-step SARSA",
                "alpha": item[1],
                "gamma": item[2],
                "epsilon": item[3],
                "n_games": n_games,
                "n_episodes": n_episodes,
                "n_step": item[0]
            }
            for _ in range(10):
                agents_list.append(agent)

    elif agent_type == "n-step SARSA approximate" or agent_type == "NSSA":
        n_step = [2, 4, 8, 12, 16]
        alpha = [0.1, 0.2, 0.3, 0.4]
        gamma = [1.0, 0.9, 0.8, 0.7]
        epsilon = [0.01, 0.05, 0.1, 0.2]

        combination = list(itertools.product(n_step, alpha, gamma, epsilon))

        for item in combination:
            agent = {
                "type": "n-step SARSA approximate",
                "alpha": item[1],
                "gamma": item[2],
                "epsilon": item[3],
                "n_games": n_games,
                "n_episodes": n_episodes,
                "n_step": item[0]
            }
            for _ in range(10):
                agents_list.append(agent)

    elif agent_type == "SARSA lambda" or agent_type == "SL":

        n_step = [2, 4, 8, 12, 16]
        alpha = [0.1, 0.2, 0.3, 0.4]
        gamma = [1.0, 0.9, 0.8, 0.7]
        epsilon = [0.01, 0.05, 0.1, 0.2]
        lambd = [0.85, 0.9, 0.92]

        combination = list(itertools.product(n_step, alpha, gamma, epsilon, lambd))

        for item in combination:
            agent = {
                "type": "SARSA lambda",
                "alpha": item[1],
                "gamma": item[2],
                "epsilon": item[3],
                "lambd": item[4],
                "n_games": n_games,
                "n_episodes": n_episodes,
                "n_step": item[0]
            }
            for _ in range(10):
                agents_list.append(agent)

    return agents_list


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


def create_agent_params_string(agent):

    if agent["type"] == "MonteCarlo" or agent["type"] == "MC":
        return "Epsilon= " + str(agent["epsilon"]) + ", gamma= " + \
            str(agent["gamma"]) + ", n_games= " + str(agent["n_games"]) + \
            ", n_episodes= " + str(agent["n_episodes"])

    elif agent["type"] == "Dynamic programming" or agent["type"] == "DP":
        return "Gamma= " + str(agent["gamma"]) + ", theta= " + str(agent["theta"])

    elif agent["type"] == "Q learning" or agent["type"] == "QL":
        return "Alpha= " + str(agent["alpha"]) + ", gamma= " + \
            str(agent["gamma"]) + ", epsilon= " + str(agent["epsilon"]) + \
            ", n_games= " + str(agent["n_games"]) + ", n_episodes= " + \
            str(agent["n_episodes"])

    elif agent["type"] == "n-step SARSA" or agent["type"] == "NSS":
        return "N-step= " + str(agent["n_step"]) + ", alpha= " + \
            str(agent["alpha"]) + ", gamma= " + str(agent["gamma"]) + ", epsilon= " \
            + str(agent["epsilon"]) + ", n_games= " + str(agent["n_games"]) + \
            ", n_episodes= " + str(agent["n_episodes"])

    elif agent["type"] == "n-step SARSA approximate" or agent["type"] == "NSSA":
        return "N-step= " + str(agent["n_step"]) + ",alpha= " + str(agent["alpha"]) + \
            ", gamma= " + str(agent["gamma"]) + ", epsilon= " + str(agent["epsilon"]) + \
            ", n_games= " + str(agent["n_games"]) + ", n_episodes= " + \
            str(agent["n_episodes"])

    elif agent["type"] == "SARSA lambda" or agent["type"] == "SL":
        return "N-step= " + str(agent["n_step"]) + \
            ",alpha= " + str(agent["alpha"]) + ", gamma= " + str(agent["gamma"]) + \
            ", epsilon= " + str(agent["epsilon"]) + ", lambda= " + \
            str(agent["lambd"]) + ", n_games= " + str(agent["n_games"]) + \
            ", n_episodes= " + str(agent["n_episodes"])

    return None


def run_agent(agent_dict):
    if agent_dict["type"] == "MonteCarlo" or agent_dict["type"] == "MC":

        dict_result = MC.run_agent(
            enviroment,
            tests_moment,
            agent_dict["n_games"],
            agent_dict["n_episodes"],
            epsilon=agent_dict["epsilon"],
            gamma=agent_dict["gamma"]
        )

    elif agent_dict["type"] == "Dynamic programming" or agent_dict["type"] == "DP":

        dict_result = DP.run_agent(
            enviroment,
            tests_moment,
            gamma=agent_dict["gamma"],
            theta=agent_dict["theta"]
        )

    elif agent_dict["type"] == "Q learning" or agent_dict["type"] == "QL":

        dict_result = QL.run_agent(
            enviroment,
            tests_moment,
            agent_dict["n_games"],
            agent_dict["n_episodes"],
            alpha=agent_dict["alpha"],
            gamma=agent_dict["gamma"],
            epsilon=agent_dict["epsilon"]

        )

    elif agent_dict["type"] == "n-step SARSA" or agent_dict["type"] == "NSS":

        dict_result = NSS.run_agent(
            enviroment,
            tests_moment,
            agent_dict["n_games"],
            agent_dict["n_episodes"],
            alpha=agent_dict["alpha"],
            gamma=agent_dict["gamma"],
            epsilon=agent_dict["epsilon"],
            n_step=agent_dict["n_step"]
        )

    elif agent_dict["type"] == "n-step SARSA approximate" or agent_dict["type"] == "NSSA":

        dict_result = NSSA.run_agent(
            enviroment,
            tests_moment,
            agent_dict["n_games"],
            agent_dict["n_episodes"],
            alpha=agent_dict["alpha"],
            gamma=agent_dict["gamma"],
            epsilon=agent_dict["epsilon"],
            n_step=agent_dict["n_step"]
        )

    elif agent_dict["type"] == "SARSA lambda" or agent_dict["type"] == "SL":

        dict_result = SL.run_agent(
            enviroment,
            tests_moment,
            agent_dict["n_games"],
            agent_dict["n_episodes"],
            alpha=agent_dict["alpha"],
            gamma=agent_dict["gamma"],
            epsilon=agent_dict["epsilon"],
            lambd=agent_dict["lambd"],
            n_step=agent_dict["n_step"]
        )

    test_result = dict_result["tests_result"]
    return test_result


if __name__ == '__main__':

    parser = parser(prog='Saving_data_multiprocess', \
        description='Reinforcement learning training agent')
    parser.add_argument('-a', '--automatic', action='store_true', \
        help='Generate automatically test for agent in enviroment')
    args = parser.parse_args()

    agents_list = []
    tests_result = []
    create_custom_enviroment()


    enviroment_name = input("Insert the enviroment name: ")
    enviroment = gym.make(enviroment_name) #Creazione ambiente
    tests_moment = input("Select the test type (final, on_run, ten_perc): ")

    #Generazione automatica degli agenti
    if args.automatic:
        agent_type = input("Insert the agent type: ")
        if (agent_type == "Dynamic programming" or agent_type == "DP") and tests_moment != "final":
            print("Dynamic Programming agent can't have on_run or ten_perc test")
            raise

        n_games = int(input("Insert the number of games: "))
        n_episodes = int(input("Insert the number of episodes for each game: "))

        agents_list = automatic_agent_generation(agent_type, n_games, n_episodes)

    #Scelta manuale degli agenti
    else:
        n_agents = int(input("Insert the number of agents: "))

        for i in range(n_agents):
            agent = input_for_agent(i, tests_moment)
            #agents_list.append(agent)
            for j in range(10):
                agents_list.append(agent)


    '''
    Launch agent
    '''
    pool = Pool(len(os.sched_getaffinity(0))-1) #creo un pool di processi
    results = pool.starmap(run_agent, zip(agents_list)) #Ogni agente viene affidato ad un processo

    pool.close()
    pool.join() # attendo che tutti gli agenti abbiano terminato il training per poi proseguire


    '''
    Create path and saving results
    '''

    base_path = "docs/" + enviroment_name + "/" + tests_moment

    for agent in range(len(agents_list)):

        complete_path = base_path + "/" + agents_list[agent]["type"] + "/"
        file = complete_path + create_agent_params_string(agents_list[agent])

        #Creo la cartella se non esiste
        if not os.path.exists(os.path.dirname(complete_path)):
            try:
                os.makedirs(os.path.dirname(complete_path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        #Se non esiste ancora creo il file nuovo con una lista vuota
        if not os.path.isfile(file):

            tmp = []

            with open(file, 'w') as outfile:
                json.dump(tmp, outfile, indent=4)


        #Appendo alla lista presente i nuovi risultati dei test
        with open(file, "r") as inputfile:
            content = json.load(inputfile)

        content.append(results[agent])

        with open(file, 'w') as outfile:
            json.dump(content, outfile, indent=4)
