import importlib

def env_choose(env):

    if not isinstance(env, str):
        env_name = env.unwrapped.spec.id
    else:
        env_name = env


    if  env_name == "FrozenLake-v0" or env_name == "FrozenLake8x8-v0" or \
        env_name == "FrozenLakeNotSlippery8x8-v0" or env_name == "FrozenLakeNotSlippery4x4-v0":

        return importlib.import_module("frozen_lake")

    elif  env_name == "Taxi-v2":

        return importlib.import_module("taxi")

    elif  env_name == "Roulette-v0":

        return importlib.import_module("roulette")

    elif  env_name == "Blackjack-v0":

        return importlib.import_module("blackjack")

    elif  env_name == "NChain-v0":

        return importlib.import_module("nchain")

    elif  env_name == "MountainCar-v0":

        return importlib.import_module("mountain_car")

    elif  env_name == "Acrobot-v1":

        return importlib.import_module("acrobot")

    elif  env_name == "CartPole-v1":

        return importlib.import_module("cartpole")

    elif  env_name == "LunarLander-v2":

        return importlib.import_module("lunar_lander")

    elif  env_name == "Pong-ram-v0":

        return importlib.import_module("pong_atari")

    elif  env_name == "MsPacman-ram-v0":

        return importlib.import_module("pacman_atari")

    return None
