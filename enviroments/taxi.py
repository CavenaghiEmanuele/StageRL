def run_game(env, action, n_action=0):

    next_state, reward, done, info = env.step(action)

    '''
    Make env more difficult (MOD1)
    '''
    if reward == -1:
        reward = 0

    return next_state, reward, done, info


def test_policy(env, action, n_action=0):

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

def reset_env(env):
    return env.reset()

def number_states(env):
    return list(range(0, env.observation_space.n))


def number_actions(env):
    return env.action_space.n


def probability(env):
    return env.env.P
