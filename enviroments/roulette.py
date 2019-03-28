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

    if done and reward >= 1:
        return {"env_info": env_info, "average": reward, "%wins": 1, "%loss": 0, "%walking away": 0}

    if done and reward == 0:
        return {"env_info": env_info, "average": reward, "%wins": 0, "%loss": 0, "%walking away": 1}

    if done and reward == -1:
        return {"env_info": env_info, "average": reward, "%wins": 0, "%loss": 1, "%walking away": 0}


    return {"env_info": env_info, "average": reward, "%wins": 0, "%loss": 0, "%walking away": 0}



def type_test():

    return ["average", "%wins", "%loss", "%walking away"]


def number_states(env):
    return list(range(0, 38)) #numero di stati


def number_actions(env, can_walking_away=True):

    action_space = 0
    if can_walking_away:
        action_space = 38 #Azioni disponibili
    else:
        action_space = 37 #Azioni disponibili

    return action_space

def reset_env(env):
    return env.reset()

def probability(env):
    return None
