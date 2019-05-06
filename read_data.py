import sys
import os
import re
import json
from math import pow, sqrt
import matplotlib.pyplot as plt
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QDialog
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi


class MonteCarloItem(QWidget):
    def __init__(self, agent):
        super(MonteCarloItem, self).__init__()
        loadUi("GUI/MonteCarloItem.ui", self)

        self.type.setText("MonteCarlo")
        self.epsilon.setText(agent["epsilon"])
        self.gamma.setText(agent["gamma"])
        self.n_games.setText(agent["n_games"])
        self.n_episodes.setText(agent["n_episodes"])

class DynamicProgrammingItem(QWidget):
    def __init__(self, agent):
        super(DynamicProgrammingItem, self).__init__()
        loadUi("GUI/DynamicProgrammingItem.ui", self)

        self.type.setText("Dynamic programming")
        self.gamma.setText(agent["gamma"])
        self.theta.setText(agent["theta"])

class QLearningItem(QWidget):
    def __init__(self, agent):
        super(QLearningItem, self).__init__()
        loadUi("GUI/QLearningItem.ui", self)

        self.type.setText("Q-Learning")
        self.alpha.setText(agent["alpha"])
        self.gamma.setText(agent["gamma"])
        self.epsilon.setText(agent["epsilon"])
        self.n_games.setText(agent["n_games"])
        self.n_episodes.setText(agent["n_episodes"])

class NStepSarsaItem(QWidget):
    def __init__(self, agent):
        super(NStepSarsaItem, self).__init__()
        loadUi("GUI/NStepSarsaItem.ui", self)

        self.type.setText("n-step SARSA")
        self.n_step.setText(agent["n_step"])
        self.alpha.setText(agent["alpha"])
        self.gamma.setText(agent["gamma"])
        self.epsilon.setText(agent["epsilon"])
        self.n_games.setText(agent["n_games"])
        self.n_episodes.setText(agent["n_episodes"])

class NStepSarsaApproximateItem(QWidget):
    def __init__(self, agent):
        super(NStepSarsaApproximateItem, self).__init__()
        loadUi("GUI/NStepSarsaApproximateItem.ui", self)

        self.type.setText("n-step SARSA approximate")
        self.n_step.setText(agent["n_step"])
        self.alpha.setText(agent["alpha"])
        self.gamma.setText(agent["gamma"])
        self.epsilon.setText(agent["epsilon"])
        self.n_games.setText(agent["n_games"])
        self.n_episodes.setText(agent["n_episodes"])

class SarsaLambdaItem(QWidget):
    def __init__(self, agent):
        super(SarsaLambdaItem, self).__init__()
        loadUi("GUI/SarsaLambdaItem.ui", self)

        self.type.setText("SARSA lambda")
        self.n_step.setText(agent["n_step"])
        self.alpha.setText(agent["alpha"])
        self.gamma.setText(agent["gamma"])
        self.epsilon.setText(agent["epsilon"])
        self.lambd.setText(agent["lambd"])
        self.n_games.setText(agent["n_games"])
        self.n_episodes.setText(agent["n_episodes"])



class AppWindow(QDialog):
    def __init__(self):
        super(AppWindow, self).__init__()
        loadUi("GUI/gui.ui", self)
        self.setWindowTitle("Read data GUI")

        self.add_enviroment.clicked.connect(self.add_enviroment_clicked)
        self.add_tests_moment.clicked.connect(self.add_tests_moment_clicked)

        self.add_mc_to_graph.clicked.connect(self.add_mc_to_graph_clicked)
        self.add_dp_to_graph.clicked.connect(self.add_dp_to_graph_clicked)
        self.add_ql_to_graph.clicked.connect(self.add_ql_to_graph_clicked)
        self.add_nss_to_graph.clicked.connect(self.add_nss_to_graph_clicked)
        self.add_nssa_to_graph.clicked.connect(self.add_nssa_to_graph_clicked)
        self.add_sl_to_graph.clicked.connect(self.add_sl_to_graph_clicked)




        self.delete_current_item.clicked.connect(self.delete_current_item_clicked)
        self.show_graph.clicked.connect(self.show_graph_clicked)


    @pyqtSlot()
    def add_enviroment_clicked(self):
        del agent_list[:]
        self.agent_list_recap.clear()

        if self.enviroment_name.currentText() != "Select enviroment":
            self.enviroment_name_recap.setText(self.enviroment_name.currentText())
            if self.tests_moment_recap.text() != "":
                self.agent_tab_widget.setEnabled(True)


    @pyqtSlot()
    def add_tests_moment_clicked(self):
        del agent_list[:]
        self.agent_list_recap.clear()

        if self.tests_moment.currentText() != "Select tests moment":
            self.tests_moment_recap.setText(self.tests_moment.currentText())
            if self.enviroment_name_recap.text() != "":
                self.agent_tab_widget.setEnabled(True)



    @pyqtSlot()
    def add_mc_to_graph_clicked(self):

        env_name = self.enviroment_name_recap.text()
        tests_moment = self.tests_moment_recap.text()
        n_games = self.mc_n_games.text()
        n_episodes = self.mc_n_episodes.text()
        epsilon = [self.mc_epsilon.text()]
        gamma = [self.mc_gamma.text()]

        if epsilon[0] == "all" and gamma[0] == "all":
            print("Only one parameter can be \"all\"")

        else:

            if epsilon[0] == "all":
                epsilon = get_all_parameter("MonteCarlo", "Epsilon", env_name, tests_moment)

            elif gamma[0] == "all":
                gamma = get_all_parameter("MonteCarlo", "gamma", env_name, tests_moment)

            for e in epsilon:
                for g in gamma:
                    agent = {
                        "type": "MonteCarlo",
                        "n_games": n_games,
                        "n_episodes": n_episodes,
                        "epsilon": e,
                        "gamma": g
                    }
                    agent_list.append(agent)
                    Item = QtWidgets.QListWidgetItem(self.agent_list_recap)
                    Item_Widget = MonteCarloItem(agent)
                    Item.setSizeHint(Item_Widget.sizeHint())
                    self.agent_list_recap.addItem(Item)
                    self.agent_list_recap.setItemWidget(Item, Item_Widget)

    @pyqtSlot()
    def add_dp_to_graph_clicked(self):

        env_name = self.enviroment_name_recap.text()
        tests_moment = self.tests_moment_recap.text()
        theta = [self.dp_theta.text()]
        gamma = [self.dp_gamma.text()]

        if theta[0] == "all" and gamma[0] == "all":
            print("Only one parameter can be \"all\"")

        else:

            if theta[0] == "all":
                theta = get_all_parameter("Dynamic programming", "theta", env_name, tests_moment)

            elif gamma[0] == "all":
                gamma = get_all_parameter("Dynamic programming", "Gamma", env_name, tests_moment)

            for t in theta:
                for g in gamma:
                    agent = {
                        "type": "Dynamic programming",
                        "gamma": g,
                        "theta": t
                    }
                    agent_list.append(agent)
                    Item = QtWidgets.QListWidgetItem(self.agent_list_recap)
                    Item_Widget = DynamicProgrammingItem(agent)
                    Item.setSizeHint(Item_Widget.sizeHint())
                    self.agent_list_recap.addItem(Item)
                    self.agent_list_recap.setItemWidget(Item, Item_Widget)


    @pyqtSlot()
    def add_ql_to_graph_clicked(self):

        env_name = self.enviroment_name_recap.text()
        tests_moment = self.tests_moment_recap.text()
        n_games = self.ql_n_games.text()
        n_episodes = self.ql_n_episodes.text()
        alpha = [self.ql_alpha.text()]
        gamma = [self.ql_gamma.text()]
        epsilon = [self.ql_epsilon.text()]

        if alpha[0] == "all" and gamma[0] == "all" and epsilon[0] == "all":
            print("Only one parameter can be \"all\"")

        elif alpha[0] == "all" and gamma[0] == "all":
            print("Only one parameter can be \"all\"")

        elif alpha[0] == "all" and epsilon[0] == "all":
            print("Only one parameter can be \"all\"")

        elif gamma[0] == "all" and epsilon[0] == "all":
            print("Only one parameter can be \"all\"")

        else:
            if alpha[0] == "all":
                alpha = get_all_parameter("Q learning", "Alpha", env_name, tests_moment)

            elif gamma[0] == "all":
                gamma = get_all_parameter("Q learning", "gamma", env_name, tests_moment)

            elif epsilon[0] == "all":
                epsilon = get_all_parameter("Q learning", "epsilon", env_name, tests_moment)

            for a in alpha:
                for g in gamma:
                    for e in epsilon:
                        agent = {
                            "type": "Q learning",
                            "alpha": a,
                            "gamma": g,
                            "epsilon": e,
                            "n_games": n_games,
                            "n_episodes": n_episodes
                        }
                        agent_list.append(agent)
                        Item = QtWidgets.QListWidgetItem(self.agent_list_recap)
                        Item_Widget = QLearningItem(agent)
                        Item.setSizeHint(Item_Widget.sizeHint())
                        self.agent_list_recap.addItem(Item)
                        self.agent_list_recap.setItemWidget(Item, Item_Widget)

    @pyqtSlot()
    def add_nss_to_graph_clicked(self):

        env_name = self.enviroment_name_recap.text()
        tests_moment = self.tests_moment_recap.text()
        n_games = self.nss_n_games.text()
        n_episodes = self.nss_n_episodes.text()
        alpha = [self.nss_alpha.text()]
        gamma = [self.nss_gamma.text()]
        epsilon = [self.nss_epsilon.text()]
        n_step = [self.nss_n_step.text()]


        if alpha[0] == "all":
            alpha = get_all_parameter("n-step SARSA", "alpha", env_name, tests_moment)

        elif gamma[0] == "all":
            gamma = get_all_parameter("n-step SARSA", "gamma", env_name, tests_moment)

        elif epsilon[0] == "all":
            epsilon = get_all_parameter("n-step SARSA", "epsilon", env_name, tests_moment)

        elif n_step[0] == "all":
            n_step = get_all_parameter("n-step SARSA", "N-step", env_name, tests_moment)

        for a in alpha:
            for g in gamma:
                for e in epsilon:
                    for n in n_step:
                        agent = {
                            "type": "n-step SARSA",
                            "alpha": a,
                            "gamma": g,
                            "epsilon": e,
                            "n_games": n_games,
                            "n_episodes": n_episodes,
                            "n_step": n
                        }
                        agent_list.append(agent)
                        Item = QtWidgets.QListWidgetItem(self.agent_list_recap)
                        Item_Widget = NStepSarsaItem(agent)
                        Item.setSizeHint(Item_Widget.sizeHint())
                        self.agent_list_recap.addItem(Item)
                        self.agent_list_recap.setItemWidget(Item, Item_Widget)


    @pyqtSlot()
    def add_nssa_to_graph_clicked(self):

        env_name = self.enviroment_name_recap.text()
        tests_moment = self.tests_moment_recap.text()
        n_games = self.nssa_n_games.text()
        n_episodes = self.nssa_n_episodes.text()
        alpha = [self.nssa_alpha.text()]
        gamma = [self.nssa_gamma.text()]
        epsilon = [self.nssa_epsilon.text()]
        n_step = [self.nssa_n_step.text()]


        if alpha[0] == "all":
            alpha = get_all_parameter("n-step SARSA approximate", "alpha", \
                env_name, tests_moment)

        elif gamma[0] == "all":
            gamma = get_all_parameter("n-step SARSA approximate", "gamma", \
                env_name, tests_moment)

        elif epsilon[0] == "all":
            epsilon = get_all_parameter("n-step SARSA approximate", "epsilon", \
                env_name, tests_moment)

        elif n_step[0] == "all":
            n_step = get_all_parameter("n-step SARSA approximate", "N-step", \
                env_name, tests_moment)

        for a in alpha:
            for g in gamma:
                for e in epsilon:
                    for n in n_step:
                        agent = {
                            "type": "n-step SARSA approximate",
                            "alpha": a,
                            "gamma": g,
                            "epsilon": e,
                            "n_games": n_games,
                            "n_episodes": n_episodes,
                            "n_step": n
                        }
                        agent_list.append(agent)
                        Item = QtWidgets.QListWidgetItem(self.agent_list_recap)
                        Item_Widget = NStepSarsaApproximateItem(agent)
                        Item.setSizeHint(Item_Widget.sizeHint())
                        self.agent_list_recap.addItem(Item)
                        self.agent_list_recap.setItemWidget(Item, Item_Widget)


    @pyqtSlot()
    def add_sl_to_graph_clicked(self):

        env_name = self.enviroment_name_recap.text()
        tests_moment = self.tests_moment_recap.text()
        n_games = self.sl_n_games.text()
        n_episodes = self.sl_n_episodes.text()
        alpha = [self.sl_alpha.text()]
        gamma = [self.sl_gamma.text()]
        epsilon = [self.sl_epsilon.text()]
        lambd = [self.sl_lambda.text()]
        n_step = [self.sl_n_step.text()]


        if alpha[0] == "all":
            alpha = get_all_parameter("SARSA lambda", "alpha", \
                env_name, tests_moment)

        elif gamma[0] == "all":
            gamma = get_all_parameter("SARSA lambda", "gamma", \
                env_name, tests_moment)

        elif epsilon[0] == "all":
            epsilon = get_all_parameter("SARSA lambda", "epsilon", \
                env_name, tests_moment)

        elif lambd[0] == "all":
            lambd = get_all_parameter("SARSA lambda", "lambda", \
                env_name, tests_moment)

        elif n_step[0] == "all":
            n_step = get_all_parameter("SARSA lambda", "N-step", \
                env_name, tests_moment)

        for a in alpha:
            for g in gamma:
                for e in epsilon:
                    for l in lambd:
                        for n in n_step:
                            agent = {
                                "type": "SARSA lambda",
                                "alpha": a,
                                "gamma": g,
                                "epsilon": e,
                                "lambd": l,
                                "n_games": n_games,
                                "n_episodes": n_episodes,
                                "n_step": n
                            }
                            agent_list.append(agent)
                            Item = QtWidgets.QListWidgetItem(self.agent_list_recap)
                            Item_Widget = SarsaLambdaItem(agent)
                            Item.setSizeHint(Item_Widget.sizeHint())
                            self.agent_list_recap.addItem(Item)
                            self.agent_list_recap.setItemWidget(Item, Item_Widget)

    @pyqtSlot()
    def delete_current_item_clicked(self):

        for selectedItem in self.agent_list_recap.selectedItems():
            del agent_list[self.agent_list_recap.row(selectedItem)]
            self.agent_list_recap.takeItem(self.agent_list_recap.row(selectedItem))


    @pyqtSlot()
    def show_graph_clicked(self):

        base_path = "docs/" + self.enviroment_name_recap.text() + "/" + \
            self.tests_moment_recap.text() + "/"
        legends = {}


        for agent in agent_list:

            path = base_path + agent["type"] + "/" + create_agent_params_string(agent)

            with open(path) as inputfile:
                tests_list = json.load(inputfile)


            for test_type in tests_list[0]:
                if not test_type in legends:
                    legends.update({test_type: list()})

                plt.figure(test_type)
                plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
                plt.ylabel(test_type)
                '''
                HOW TO SHOW RESULT HERE
                '''
                if self.how_result.currentText() == "All results":

                    for i_test_agent in range(len(tests_list)):
                        plt.plot(tests_list[i_test_agent][test_type],  \
                            label=create_legend_string(agent))
                        plt.legend(loc='upper left')


                elif self.how_result.currentText() == "Average results":

                    average = []
                    for _ in tests_list[0][test_type]:
                        average.append(0)

                    for test_of_i_agent in tests_list:
                        for i in range(len(test_of_i_agent[test_type])):
                            average[i] += test_of_i_agent[test_type][i]

                    for i in range(len(average)):
                        average[i] = average[i] / len(tests_list)

                    plt.plot(average, label=create_legend_string(agent))
                    plt.legend(loc='upper left')


                    plt.figure("Standard deviation of " + test_type)
                    plt.ylabel(test_type)
                    plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)

                    standard_deviation = []
                    for _ in tests_list[0][test_type]:
                        standard_deviation.append(0)

                    for test_of_i_agent in tests_list:
                        for i in range(len(test_of_i_agent[test_type])):
                            standard_deviation[i] += pow(test_of_i_agent[test_type][i] \
                                - average[i], 2)

                    for i in range(len(standard_deviation)):
                        standard_deviation[i] = sqrt(standard_deviation[i] / \
                            len(tests_list))

                    plt.plot(standard_deviation, label=create_legend_string(agent))
                    plt.legend(loc='upper left')

        plt.show()




def create_agent_params_string(agent):

    if agent["type"] == "MonteCarlo" or agent["type"] == "MC":
        return "Epsilon= " + str(agent["epsilon"]) + ", gamma= " + \
            str(agent["gamma"]) + ", n_games= " + str(agent["n_games"]) + \
            ", n_episodes= " + str(agent["n_episodes"])

    elif agent["type"] == "Dynamic programming" or agent["type"] == "DP":
        return "Gamma= " + str(agent["gamma"]) + ", theta= " + str(agent["theta"])

    elif agent["type"] == "Q learning" or agent["type"] == "QL":
        return "Alpha= " + str(agent["alpha"]) + ", gamma= " + str(agent["gamma"]) \
            + ", epsilon= " + str(agent["epsilon"]) + ", n_games= " + \
            str(agent["n_games"]) + ", n_episodes= " + str(agent["n_episodes"])

    elif agent["type"] == "n-step SARSA" or agent["type"] == "NSS":
        return "N-step= " + str(agent["n_step"]) + ", alpha= " + str(agent["alpha"]) \
            + ", gamma= " + str(agent["gamma"]) + ", epsilon= " + str(agent["epsilon"]) \
            + ", n_games= " + str(agent["n_games"]) + ", n_episodes= " + \
            str(agent["n_episodes"])

    elif agent["type"] == "n-step SARSA approximate" or agent["type"] == "NSSA":
        return "N-step= " + str(agent["n_step"]) + ",alpha= " + str(agent["alpha"])\
            + ", gamma= " + str(agent["gamma"]) + ", epsilon= " + str(agent["epsilon"]) \
            + ", n_games= " + str(agent["n_games"]) + ", n_episodes= " + \
            str(agent["n_episodes"])

    elif agent["type"] == "SARSA lambda" or agent["type"] == "SL":
        return "N-step= " + str(agent["n_step"]) + \
            ",alpha= " + str(agent["alpha"]) + ", gamma= " + str(agent["gamma"]) + \
            ", epsilon= " + str(agent["epsilon"]) + ", lambda= " + \
            str(agent["lambd"]) + ", n_games= " + str(agent["n_games"]) + \
            ", n_episodes= " + str(agent["n_episodes"])

    return None


def create_legend_string(agent):

    if agent["type"] == "MonteCarlo" or agent["type"] == "MC":
        return "MonteCarlo, epsilon= " + str(agent["epsilon"]) + ", gamma= " \
            + str(agent["gamma"]) + ", n_games= " + str(agent["n_games"]) + \
            ", n_episodes= " + str(agent["n_episodes"])

    elif agent["type"] == "Dynamic programming" or agent["type"] == "DP":
        return "Dynamic programming, gamma= " + str(agent["gamma"]) + ", theta= " \
            + str(agent["theta"])

    elif agent["type"] == "Q learning" or agent["type"] == "QL":
        return "Q learning, alpha= " + str(agent["alpha"]) + ", gamma= " + \
            str(agent["gamma"]) + ", epsilon= " + str(agent["epsilon"]) + \
            ", n_games= " + str(agent["n_games"]) + ", n_episodes= " + \
            str(agent["n_episodes"])

    elif agent["type"] == "n-step SARSA" or agent["type"] == "NSS":
        return "n-step SARSA, n-step= " + str(agent["n_step"]) + ",alpha= " + \
            str(agent["alpha"]) + ", gamma= " + str(agent["gamma"]) + ", epsilon= "\
            + str(agent["epsilon"]) + ", n_games= " + str(agent["n_games"]) \
            + ", n_episodes= " + str(agent["n_episodes"])

    elif agent["type"] == "n-step SARSA approximate" or agent["type"] == "NSSA":
        return "n-step SARSA approximate, n-step= " + str(agent["n_step"]) + \
            ",alpha= " + str(agent["alpha"]) + ", gamma= " + str(agent["gamma"]) \
            + ", epsilon= " + str(agent["epsilon"]) + ", n_games= " + \
            str(agent["n_games"]) + ", n_episodes= " + str(agent["n_episodes"])

    elif agent["type"] == "SARSA lambda" or agent["type"] == "SL":
        return "SARSA lambda, n-step= " + str(agent["n_step"]) + \
            ",alpha= " + str(agent["alpha"]) + ", gamma= " + str(agent["gamma"]) + \
            ", epsilon= " + str(agent["epsilon"]) + ", lambda= " + \
            str(agent["lambd"]) + ", n_games= " + str(agent["n_games"]) + \
            ", n_episodes= " + str(agent["n_episodes"])

    return None

def get_all_parameter(agent_type, parameter, env_name, tests_moment_name):

    path = "docs/" + env_name + "/" + tests_moment_name + "/" + agent_type

    parameter_set = set()
    for file in os.listdir(path):
        s = re.search(parameter + "= (\d+\.\d+)", file)
        if s == None: #For DP Agent
            s = re.search(parameter + "= (1e-\d+)", file)
        if s == None: #for NSS Agent
            s = re.search(parameter + "= (\d+)", file)

        parameter_set.add(s.group(1))

    return parameter_set


if __name__ == '__main__':

    agent_list = []

    enviroment_name = ""
    tests_moment = ""

    app = QApplication(sys.argv)
    w = AppWindow()
    w.show()

    sys.exit(app.exec_())
