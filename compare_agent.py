import sys
import os
import re
import json
import heapq
import numpy as np
from math import pow, sqrt
import matplotlib.pyplot as plt
sys.path.insert(0, 'enviroments')
import enviroment_choose


def create_legend(agent_type, filename):
    return agent_type + ", " + filename


def average_of_agents_test(tests_i_agent):

    average_i_agent = [0]*len(tests_i_agent[0][test_type])

    for one_test_i_agent in tests_i_agent:
        for i in range(len(one_test_i_agent[test_type])):
            average_i_agent[i] += one_test_i_agent[test_type][i]

    for i in range(len(average_i_agent)):
        average_i_agent[i] = average_i_agent[i] / len(tests_i_agent)

    return average_i_agent


def percentile_10_of_agents_test(tests_i_agent):

    tmp = np.zeros((len(tests_i_agent[0][test_type]), len(tests_i_agent)))

    for i in range(len(tests_i_agent)):
        for j in range(len(tests_i_agent[i][test_type])):
            tmp[j][i] = tests_i_agent[i][test_type][j]

    percentile_10 = []
    for i in range(len(tmp)):
        percentile_10.append(np.percentile(tmp[i], 10))

    return percentile_10


def quartile_1_of_agents_test(tests_i_agent):

    tmp = np.zeros((len(tests_i_agent[0][test_type]), len(tests_i_agent)))

    for i in range(len(tests_i_agent)):
        for j in range(len(tests_i_agent[i][test_type])):
            tmp[j][i] = tests_i_agent[i][test_type][j]

    quartile_1 = []
    for i in range(len(tmp)):
        quartile_1.append(np.percentile(tmp[i], 25))

    return quartile_1


def median_of_agents_test(tests_i_agent):

    tmp = np.zeros((len(tests_i_agent[0][test_type]), len(tests_i_agent)))

    for i in range(len(tests_i_agent)):
        for j in range(len(tests_i_agent[i][test_type])):
            tmp[j][i] = tests_i_agent[i][test_type][j]

    median = []
    for i in range(len(tmp)):
        median.append(np.median(tmp[i]))

    return median


def quartile_3_of_agents_test(tests_i_agent):

    tmp = np.zeros((len(tests_i_agent[0][test_type]), len(tests_i_agent)))

    for i in range(len(tests_i_agent)):
        for j in range(len(tests_i_agent[i][test_type])):
            tmp[j][i] = tests_i_agent[i][test_type][j]

    quartile_3 = []
    for i in range(len(tmp)):
        quartile_3.append(np.percentile(tmp[i], 75))

    return quartile_3


def percentile_90_of_agents_test(tests_i_agent):

    tmp = np.zeros((len(tests_i_agent[0][test_type]), len(tests_i_agent)))

    for i in range(len(tests_i_agent)):
        for j in range(len(tests_i_agent[i][test_type])):
            tmp[j][i] = tests_i_agent[i][test_type][j]

    percentile_90 = []
    for i in range(len(tmp)):
        percentile_90.append(np.percentile(tmp[i], 90))

    return percentile_90



if __name__ == '__main__':

    env_name = input("Insert the enviroment name: ")
    env = enviroment_choose.env_choose(env_name)
    tests_moment = input("Select the test type (final, on_run, ten_perc): " )
    how_group_same_agent = input("Select how group the results of same agent \n" + \
        "(Average, 10th percentile, Quartile 1, Median, Quartile 3, 90th percentile): ")
    number_of_agent_for_type = int(input("Insert the number of best agent for every type of agent: "))
    base_path = "docs/" + env_name + "/" + tests_moment + "/"


    all_agent_tests = {}
    all_agent_legend = {}
    agent_type_list = [x[1] for x in os.walk(base_path)]


    for agent_type in agent_type_list[0]:

        path = base_path + agent_type

        ### Costruisco i dizionari ###
        all_agent_tests.update({agent_type: dict()})
        all_agent_legend.update({agent_type: dict()})
        for test_type in env.type_test():
            all_agent_tests[agent_type].update({test_type: list()})
            all_agent_legend[agent_type].update({test_type: list()})
        ##############

        for filename in os.listdir(path):
            complete_path = path + "/" + filename
            with open(complete_path) as inputfile:
                tests_i_agent = json.load(inputfile)


            for test_type in tests_i_agent[0]:

                ### Raggruppamento per media
                if how_group_same_agent == "Average":
                    all_agent_tests[agent_type][test_type].append(average_of_agents_test(tests_i_agent))

                elif how_group_same_agent == "10th percentile":
                    all_agent_tests[agent_type][test_type].append(percentile_10_of_agents_test(tests_i_agent))

                elif how_group_same_agent == "Quartile 1":
                    all_agent_tests[agent_type][test_type].append(quartile_1_of_agents_test(tests_i_agent))

                elif how_group_same_agent == "Median":
                    all_agent_tests[agent_type][test_type].append(median_of_agents_test(tests_i_agent))

                elif how_group_same_agent == "Quartile 3":
                    all_agent_tests[agent_type][test_type].append(quartile_3_of_agents_test(tests_i_agent))

                elif how_group_same_agent == "90th percentile":
                    all_agent_tests[agent_type][test_type].append(percentile_90_of_agents_test(tests_i_agent))


                all_agent_legend[agent_type][test_type].append(create_legend(agent_type, filename))



    ### Trovo l'agente migliore per ogni tipo di agente usando la media dei valori
    ### di ciascun agente come valore per il confronto
    bests_agent = []

    for agent_type in all_agent_tests:
        for test_type in all_agent_tests[agent_type]:

            agent_average = []
            for count, test_agent_i in enumerate(all_agent_tests[agent_type][test_type]):

                if tests_moment == "final":
                    agent_average.append([sum(test_agent_i) / len(test_agent_i), agent_type, test_type, count])
                #Nel caso dei test ten_perc e on_run teniamo maggiormente conto
                #dei risultati man mano che si avvicinano al termine del training
                elif tests_moment == "ten_perc":
                    agent_average.append([np.average(test_agent_i, weights=np.arange(0.1, 1.1, 0.1)), agent_type, test_type, count])

                elif tests_moment == "on_run":
                    agent_average.append([np.average(test_agent_i, weights=np.arange(1, 101 )), agent_type, test_type, count])

            #best_test = [average, agent_type, test_type, number of agent in all_agent_tests]
            best_test = heapq.nlargest(number_of_agent_for_type, agent_average)

            for i in range(number_of_agent_for_type):
                bests_agent.append(
                    {
                    "agent": all_agent_tests[best_test[i][1]][best_test[i][2]][best_test[i][3]],
                    "legend": all_agent_legend[best_test[i][1]][best_test[i][2]][best_test[i][3]],
                    "average": best_test[i][0],
                    "test_type": test_type
                    }
                )



    #Ordino gli agenti migliori dal migliore al peggiore
    bests_agent.sort(key=lambda x: x["average"], reverse=True)

    for test_type in env.type_test():

        plt.figure(how_group_same_agent + " of " + test_type)
        plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
        plt.ylabel(test_type)

        for agent in bests_agent:
            if agent["test_type"] == test_type:
                plt.plot(agent["agent"], label=agent["legend"] + " " + str(agent["average"]))
                plt.legend(loc='upper left')

    plt.show()
