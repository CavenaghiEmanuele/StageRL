import random


def run_game(env, policy):
     env.reset()
     episode = []
     finished = False

     while not finished:
          state = env.env.s

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


def test_policy(policy, env, type_test="average"):

    if type_test == "average":

            reward = 0
            r = 100
            for i in range(r):
                episode = run_game(env, policy)
                for step in range(len(episode)):
                    reward += episode[step][2]

            return reward/r

    elif type_test == "%wins":

        wins = 0
        r = 1000
        for i in range(r):
            w = run_game(env, policy)[-1][-1]
            if w == 20:
                wins += 1

        return wins / r
