import sys
import os
import re
import json
import heapq
from math import pow, sqrt
import matplotlib.pyplot as plt
sys.path.insert(0, 'enviroments')
import enviroment_choose


def create_legend(agent_type, filename):
    return agent_type + ", " + filename



if __name__ == '__main__':

    env = enviroment_choose.env_choose("Taxi-v2")
    base_path = "docs/" + "Taxi-v2" + "/" + "final" + "/"


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

                average_i_agent = [0]*len(tests_i_agent[0][test_type])

                #######Raggruppo gli agenti uguali (MEDIA)###########
                for one_test_i_agent in tests_i_agent:
                    for i in range(len(one_test_i_agent[test_type])):
                        average_i_agent[i] += one_test_i_agent[test_type][i]

                for i in range(len(average_i_agent)):
                    average_i_agent[i] = average_i_agent[i] / len(tests_i_agent)
                #######################

                all_agent_tests[agent_type][test_type].append(average_i_agent)
                all_agent_legend[agent_type][test_type].append(create_legend(agent_type, filename))



    ### Trovo l'agente migliore per ogni tipo di agente
    tests_to_plot = []
    legend_to_plot = []

    for agent_type in all_agent_tests:
        for test_type in all_agent_tests[agent_type]:

            plt.figure(test_type)
            plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
            plt.ylabel(test_type)

            agent_average = []
            for count, test_agent_i in enumerate(all_agent_tests[agent_type][test_type]):
                agent_average.append([sum(test_agent_i) / len(test_agent_i), agent_type, test_type, count])


            number_of_elements = 1
            best_test = heapq.nlargest(number_of_elements, agent_average)


            for i in range(number_of_elements):
                tests_to_plot.append(all_agent_tests[best_test[i][1]][best_test[i][2]][best_test[i][3]])
                legend_to_plot.append(str(i+1) + "° " + all_agent_legend[best_test[i][1]][best_test[i][2]][best_test[i][3]])


                plt.plot(all_agent_tests[best_test[i][1]][best_test[i][2]][best_test[i][3]], label=str(i+1) + "° " + all_agent_legend[best_test[i][1]][best_test[i][2]][best_test[i][3]])
                plt.legend(loc='upper left')


    plt.show()
