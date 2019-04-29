import sys
import json
import matplotlib.pyplot as plt
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QDialog
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUi


class MonteCarloItem(QWidget):
    def __init__ (self, agent):
        super(MonteCarloItem, self).__init__()
        loadUi("GUI/MonteCarloItem.ui", self)

        self.type.setText("MonteCarlo")
        self.epsilon.setText(agent["epsilon"])
        self.gamma.setText(agent["gamma"])
        self.n_games.setText(agent["n_games"])
        self.n_episodes.setText(agent["n_episodes"])

class DynamicProgrammingItem(QWidget):
    def __init__ (self, agent):
        super(DynamicProgrammingItem, self).__init__()
        loadUi("GUI/DynamicProgrammingItem.ui", self)

        self.type.setText("Dynamic programming")
        self.gamma.setText(agent["gamma"])
        self.theta.setText(agent["theta"])

class QLearningItem(QWidget):
    def __init__ (self, agent):
        super(QLearningItem, self).__init__()
        loadUi("GUI/QLearningItem.ui", self)

        self.type.setText("Q-Learning")
        self.alpha.setText(agent["alpha"])
        self.gamma.setText(agent["gamma"])
        self.epsilon.setText(agent["epsilon"])
        self.n_games.setText(agent["n_games"])
        self.n_episodes.setText(agent["n_episodes"])

class NStepSarsaItem(QWidget):
    def __init__ (self, agent):
        super(NStepSarsaItem, self).__init__()
        loadUi("GUI/NStepSarsaItem.ui", self)

        self.type.setText("n-step SARSA")
        self.n_step.setText(agent["n_step"])
        self.alpha.setText(agent["alpha"])
        self.gamma.setText(agent["gamma"])
        self.epsilon.setText(agent["epsilon"])
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



        self.delete_current_item.clicked.connect(self.delete_current_item_clicked)
        self.show_graph.clicked.connect(self.show_graph_clicked)


    @pyqtSlot()
    def add_enviroment_clicked(self):
        del agent_list[:]
        self.agent_list_recap.clear()
        self.enviroment_name_recap.setText(self.enviroment_name.text())

    @pyqtSlot()
    def add_tests_moment_clicked(self):
        del agent_list[:]
        self.agent_list_recap.clear()
        self.tests_moment_recap.setText(self.tests_moment.text())




    @pyqtSlot()
    def add_mc_to_graph_clicked(self):

        agent ={
            "type": "MonteCarlo",
            "n_games": self.mc_n_games.text(),
            "n_episodes": self.mc_n_episodes.text(),
            "epsilon": self.mc_epsilon.text(),
            "gamma": self.mc_gamma.text()
        }
        agent_list.append(agent)
        Item = QtWidgets.QListWidgetItem(self.agent_list_recap)
        Item_Widget = MonteCarloItem(agent)
        Item.setSizeHint(Item_Widget.sizeHint())
        self.agent_list_recap.addItem(Item)
        self.agent_list_recap.setItemWidget(Item, Item_Widget)

    @pyqtSlot()
    def add_dp_to_graph_clicked(self):

        agent ={
            "type": "Dynamic programming",
            "gamma": self.dp_gamma.text(),
            "theta": self.dp_theta.text()
        }
        agent_list.append(agent)
        Item = QtWidgets.QListWidgetItem(self.agent_list_recap)
        Item_Widget = DynamicProgrammingItem(agent)
        Item.setSizeHint(Item_Widget.sizeHint())
        self.agent_list_recap.addItem(Item)
        self.agent_list_recap.setItemWidget(Item, Item_Widget)

    @pyqtSlot()
    def add_ql_to_graph_clicked(self):

        agent ={
            "type": "Q learning",
            "alpha": self.ql_alpha.text(),
            "gamma": self.ql_gamma.text(),
            "epsilon": self.ql_epsilon.text(),
            "n_games": self.ql_n_games.text(),
            "n_episodes": self.ql_n_episodes.text()
        }
        agent_list.append(agent)
        Item = QtWidgets.QListWidgetItem(self.agent_list_recap)
        Item_Widget = QLearningItem(agent)
        Item.setSizeHint(Item_Widget.sizeHint())
        self.agent_list_recap.addItem(Item)
        self.agent_list_recap.setItemWidget(Item, Item_Widget)

    @pyqtSlot()
    def add_nss_to_graph_clicked(self):

        agent ={
            "type": "n-step SARSA",
            "alpha": self.nss_alpha.text(),
            "gamma": self.nss_gamma.text(),
            "epsilon": self.nss_epsilon.text(),
            "n_games": self.nss_n_games.text(),
            "n_episodes": self.nss_n_episodes.text(),
            "n_step": self.nss_n_step.text()
        }
        agent_list.append(agent)
        Item = QtWidgets.QListWidgetItem(self.agent_list_recap)
        Item_Widget = NStepSarsaItem(agent)
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

        base_path = "docs/" + self.enviroment_name_recap.text() + "/" + self.tests_moment_recap.text() + "/"
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

                '''
                HOW TO SHOW RESULT HERE
                '''
                if self.how_result.currentText() == "All results":

                    for i_test_agent in range(len(tests_list)):
                        plt.plot(tests_list[i_test_agent][test_type])
                        legends[test_type].append(create_legend_string(agent))


                elif self.how_result.currentText() == "Average results":

                    sum = []
                    for _ in tests_list[0][test_type]:
                        sum.append(0)

                    for test_of_i_agent in tests_list:
                        for i in range(len(test_of_i_agent[test_type])):
                            sum[i] += test_of_i_agent[test_type][i]

                    for i in range(len(sum)):
                        sum[i] = sum[i] / len(tests_list)

                    plt.plot(sum)
                    legends[test_type].append(create_legend_string(agent))





                plt.ylabel(test_type)
                plt.legend(legends[test_type], loc='upper left')

        plt.show()






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

    agent_list = []

    enviroment_name = ""
    tests_moment = ""

    app = QApplication(sys.argv)
    w = AppWindow()
    w.show()

    sys.exit(app.exec_())
