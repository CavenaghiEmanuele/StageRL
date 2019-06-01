def run_game(env, action, n_action=0):

    return env.step(action)


def test_policy(env, action, n_action=0):

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


def number_states(env):
    return list(range(0, env.observation_space.n))


def number_actions(env):
    return env.action_space.n

def reset_env(env):
    return env.reset()


def probability(env):
    return None
