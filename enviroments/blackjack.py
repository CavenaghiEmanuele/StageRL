import random
import itertools as it

def run_game(env, action):

    next_state, reward, done, info = env.step(action)

    state = next_state[0] + 32*next_state[1] + 11*32*int(next_state[2])


    return


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

    '''
    list = []
    for k in range(0, 32):
        for j in range(0, 11):
            for i in [True, False]:
                list.append(0)

    tmp = [range(0, 32), range(0, 11), [True, False]]
    tmp = len(list(it.product(*tmp)))
    '''
    return list(range(0, 32*11*2))


def number_actions(env):
    return env.action_space.n


def probability(env):
    return None
