import random


def run_game(env, policy):
     env.reset()
     episode = []
     finished = False

     while not finished:
          state = env.action_space.sample()

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


def test_policy(policy, env):

    reward = 0
    r = 1000
    for i in range(r):
        episode = run_game(env, policy)
        for step in range(len(episode)):
            reward += episode[step][2]

    return reward/r


def number_states(env):
    return list(range(0, env.observation_space.n))


def number_actions(env):
    return env.action_space.n
