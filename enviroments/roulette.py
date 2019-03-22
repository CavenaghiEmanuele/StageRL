import random


def run_game(env, policy):
     env.reset()
     episode = []
     finished = False

     state = env.action_space.sample()

     while not finished:

          timestep = []
          timestep.append(state)
          n = random.uniform(0, sum(policy[state].values()))
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
            r = 100
            for i in range(r):
                episode = run_game_for_test(env, policy)
                for step in range(len(episode)):
                    reward += episode[step][2]

            return reward/r

    elif type_test == "%wins":

        wins = 0
        r = 1000
        for i in range(r):
            w = run_game(env, policy)[-1][-1]
            if w >= 1:
                wins += 1

        return wins / r

    elif type_test == "%loss":

        loss = 0
        r = 1000
        for i in range(r):
            w = run_game(env, policy)[-1][-1]
            if w == -1:
                loss += 1

        return loss / r

    elif type_test == "%walking away":

        stop = 0
        r = 1000
        for i in range(r):
            w = run_game(env, policy)[-1][-1]
            if w == 0:
                stop += 1

        return stop / r


def number_states(env):
    return list(range(0, 38)) #numero di stati


def number_actions(env, can_walking_away=True):

    action_space = 0
    if can_walking_away:
        action_space = 38 #Azioni disponibili
    else:
        action_space = 37 #Azioni disponibili

    return action_space


def probability(env):
    return None
