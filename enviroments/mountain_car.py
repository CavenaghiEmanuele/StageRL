import numpy as np


def run_game(env, action):

    next_state, reward, done, info = env.step(action)

    state_adj = (next_state - env.observation_space.low)*np.array([10, 100])
    state_adj = np.round(state_adj, 0).astype(int)
    state_adj = state_adj[0] + 19*state_adj[1]

    return [state_adj, reward, done, info]


def test_policy(env, action):

    next_state, reward, done, info = env.step(action)
    state_adj = (next_state - env.observation_space.low)*np.array([10, 100])
    state_adj = np.round(state_adj, 0).astype(int)
    state_adj = state_adj[0] + 19*state_adj[1]

    env_info = {
        "next_state": state_adj,
        "reward": reward,
        "done": done,
        "info": info
        }

    if done and reward == 1:
        return {"env_info": env_info, "average": reward}


    return {"env_info": env_info, "average": reward}




def type_test():

    return ["average"]

def number_states(env):
    '''
    Discretize the state space. One simple way in which this can be done
    is to round the first element of the state vector to the nearest 0.1
    and the second element to the nearest 0.01, and then (for convenience)
    multiply the first element by 10 and the second by 100.
    '''
    return list(range(0, 285))


def number_actions(env):
    return env.action_space.n

def reset_env(env):

    state = env.reset()

    state_adj = (state - env.observation_space.low)*np.array([10, 100])
    state_adj = np.round(state_adj, 0).astype(int)

    return state_adj[0] + 19*state_adj[1]
