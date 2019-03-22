import random
import itertools as it

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


def run_game_for_test(env, policy):
    env.reset()
    episode = []
    finished = False

    state = env.action_space.sample()

    while not finished:

        timestep = []
        timestep.append(state)

        action_max = [0, -1]
        for prob in policy[state].items():

            if prob[1] > action_max[1]:
                action_max = prob

        action = action_max[0]

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
            episode = run_game_for_test(env, policy)
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


def number_states(env):

    tmp = [range(0, 32), range(0, 11), [True, False]]
    return list(it.product(*tmp))


def number_actions(env):
    return env.action_space.n


def probability(env):
    return None
