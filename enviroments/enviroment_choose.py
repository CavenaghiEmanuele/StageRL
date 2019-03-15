import importlib


def env_choose(env):

    env_name = env.unwrapped.spec.id

    if  env_name == "FrozenLake-v0" or env_name ==  "FrozenLake8x8-v0":

        return importlib.import_module("frozen_lake")
