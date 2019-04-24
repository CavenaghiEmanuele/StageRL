import sys
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QDialog
from PyQt5.uic import loadUi



class AppWindow(QDialog):
    def __init__(self):
        super(AppWindow, self).__init__()
        loadUi("GUI/gui.ui", self)
        self.setWindowTitle("Read data GUI")
        self.add_mc_to_graph.clicked.connect(self.add_mc_to_graph_clicked)

        self.show_graph.clicked.connect(self.show_graph_clicked)



    @pyqtSlot()
    def add_mc_to_graph_clicked(self):

        agent ={
            "type": "MonteCarlo",
            "n_games": self.mc_n_games.text(),
            "n_episodes": self.mc_n_episodes.text(),
            "epsilon": self.mc_epsilon.text(),
            "gamma": self.mc_gamma.text()
        }




    @pyqtSlot()
    def show_graph_clicked(self):

        enviroment_name = self.enviroment_name.text()
        tests_moment = self.tests_moment.text()

        path = "docs/" + enviroment_name + "/" + tests_moment + "/"
        #path = "docs/" + enviroment_name + "/" + tests_moment + "/" + agent["type"] + "/" + create_agent_params_string(agent)

        self.enviroment_name.setText(path)




if __name__ == '__main__':


    app = QApplication(sys.argv)
    w = AppWindow()
    w.show()

    sys.exit(app.exec_())





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
