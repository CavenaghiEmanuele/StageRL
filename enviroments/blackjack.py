import random

def run_game(env, policy):
     env.reset()
     episode = []
     finished = False

     state = env.reset()

     while not finished:

          timestep = []
          timestep.append(state)
          n = random.uniform(0, sum(policy[(state)].values()))
          top_range = 0
          for prob in policy[state].items():
             top_range += prob[1]
             if n < top_range:
                   action = prob[0]
                   break
          state, reward, finished, info = env.step(action)
          timestep.append(action)
          timestep.append(reward)

          episode.append(timestep)

     return episode


def test_policy(policy, env, type_test="average"):

    if type_test == "average":

        reward = 0
        r = 1000
        for i in range(r):
            episode = run_game(env, policy)
            for step in range(len(episode)):
                reward += episode[step][2]

        return reward/r


    elif type_test == "total":

        reward = 0
        r = 1000
        for i in range(r):
            episode = run_game(env, policy)
            for step in range(len(episode)):
                reward += episode[step][2]

        return reward


    elif type_test == "%wins":

        wins = 0
        r = 1000
        for i in range(r):
            w = run_game(env, policy)[-1][-1]
            if w == 1:
                wins += 1

        return wins / r

    elif type_test == "%drawing":

        wins = 0
        r = 1000
        for i in range(r):
            w = run_game(env, policy)[-1][-1]
            if w == 0:
                wins += 1

        return wins / r

    elif type_test == "%loss":

        wins = 0
        r = 1000
        for i in range(r):
            w = run_game(env, policy)[-1][-1]
            if w == -1:
                wins += 1

        return wins / r




def create_random_policy(env):
    action_space = env.action_space.n #Azioni disponibili
    policy = {}
    for i in range(0, 32):
        for j in range(0, 11):
            for k in [True, False]:
                current_end = 0
                p = {}
                for action in range(0, action_space):
                    p[action] = 1 / action_space
                policy[(i, j, k)] = p
    return policy


def create_state_action_dictionary(env, policy):
    action_space = env.action_space.n #Azioni disponibili
    Q = {}
    for key in policy.keys():
         Q[key] = {a: 0.0 for a in range(0, action_space)}
    return Q
