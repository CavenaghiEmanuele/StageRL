import gym
import numpy as np
import matplotlib.pyplot as plt
import operator
from IPython.display import clear_output
from time import sleep
import itertools
from argparse import ArgumentParser as parser

import agents.n_step_sarsa_approximate as NSSAA



if __name__ == '__main__':

    enviroment = gym.make("MountainCar-v0") #Creazione ambiente


    NSSAA.run_agent(enviroment, 10, 10)
