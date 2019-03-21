import importlib
from IPython.display import clear_output
from time import sleep

def env_choose(env):

    env_name = env.unwrapped.spec.id

    if  env_name == "FrozenLake-v0" or env_name ==  "FrozenLake8x8-v0" or env_name == "FrozenLakeNotSlippery8x8-v0" or env_name == "FrozenLakeNotSlippery4x4-v0":

        return importlib.import_module("frozen_lake")

    elif  env_name == "Taxi-v2":

        return importlib.import_module("taxi")

    elif  env_name == "Roulette-v0":

        return importlib.import_module("roulette")

    elif  env_name == "Blackjack-v0":

        return importlib.import_module("blackjack")

    elif  env_name == "NChain-v0":

        return importlib.import_module("nchain")
