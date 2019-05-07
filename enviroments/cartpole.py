import numpy as np
from agents.tiles import *



def run_game_approximate(env, action):
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
    cart_position = tiling_dim / (env.observation_space.high[0] \
                                              - env.observation_space.low[0])
    cart_velocity = tiling_dim / env.observation_space.high[1]

    pole_angle = tiling_dim / (env.observation_space.high[2] \
                                              - env.observation_space.low[2])
    pole_velocity = tiling_dim / env.observation_space.high[3]

    return [cart_position, cart_velocity, pole_angle, pole_velocity]

def num_tilings():
    return 16

def IHT_max_size():
    return 4096
