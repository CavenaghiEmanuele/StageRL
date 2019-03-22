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


def test_policy(policy, env):
    wins = 0
    r = 1000
    for i in range(r):
        w = run_game_for_test(env, policy)[-1][-1]
        if w == 1:
            wins += 1

    return wins / r


def number_states(env):
    return list(range(0, env.observation_space.n))


def number_actions(env):
    return env.action_space.n

def probability(env):
    return env.env.P
