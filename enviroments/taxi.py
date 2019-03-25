import random


def run_game(env, action):

    return env.step(action)


def test_policy(env, action):

    next_state, reward, done, info = env.step(action)

    env_info = {
        "next_state": next_state,
        "reward": reward,
        "done": done,
        "info": info
        }

    if done and reward == 20:
        return {"env_info": env_info, "average": reward, "%wins": 1}


    return {"env_info": env_info, "average": reward, "%wins": 0}


def type_test():

    return ["average", "%wins"]


def number_states(env):
    return list(range(0, env.observation_space.n))


def number_actions(env):
    return env.action_space.n


def probability(env):
    return env.env.P
