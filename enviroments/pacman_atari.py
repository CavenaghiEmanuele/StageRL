import numpy as np
from agents.tiles import *



def run_game_approximate(env, action):
    env.render()
    return env.step(action)



def test_policy_approximate(env, action):

    next_state, reward, done, info = env.step(action)

    env_info = {
        "next_state": next_state,
        "reward": reward,
        "done": done,
        "info": info
        }

    return {"env_info": env_info, "average": reward}



def type_test():

    return ["average"]


def number_actions(env):
    return env.action_space.n


def reset_env_approximate(env):
    return env.reset()

def probability(env):
    return None


def features_vector(env, tiling_dim):

    list = [tiling_dim/256]*128
    return list

def num_tilings():
    return 1024

def IHT_max_size():
    return 16384
