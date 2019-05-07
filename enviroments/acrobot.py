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
    cos_theta1 = tiling_dim / (env.observation_space.high[0] \
                                              - env.observation_space.low[0])
    sin_theta1 = tiling_dim / (env.observation_space.high[1] \
                                              - env.observation_space.low[1])
    cos_theta2 = tiling_dim / (env.observation_space.high[2] \
                                              - env.observation_space.low[2])
    sin_theta2 = tiling_dim / (env.observation_space.high[3] \
                                              - env.observation_space.low[3])
    thetaDot1 = tiling_dim / (env.observation_space.high[4] \
                                              - env.observation_space.low[4])
    thetaDot2 = tiling_dim / (env.observation_space.high[5] \
                                              - env.observation_space.low[5])

    return [cos_theta1, sin_theta1, cos_theta2, sin_theta2, thetaDot1, thetaDot2]


def num_tilings():
    return 32

def IHT_max_size():
    return 16777216
