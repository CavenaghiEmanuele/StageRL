import random
import itertools as it

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

    if done and reward == 1:
        return {"env_info": env_info, "average": reward, "%wins": 1, "%drawing": 0, "%loss": 0}

    if done and reward == 0:
        return {"env_info": env_info, "average": reward, "%wins": 0, "%drawing": 1, "%loss": 0}

    if done and reward == -1:
        return {"env_info": env_info, "average": reward, "%wins": 0, "%drawing": 0, "%loss": 1}


    return {"env_info": env_info, "average": reward, "%wins": 0, "%drawing": 0, "%loss": 0}



def type_test():

    return ["average", "%wins", "%drawing", "%loss"]



def number_states(env):

    tmp = [range(0, 32), range(0, 11), [True, False]]
    return list(it.product(*tmp))


def number_actions(env):
    return env.action_space.n


def probability(env):
    return None
