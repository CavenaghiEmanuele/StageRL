import random
from tqdm import tqdm
import sys
sys.path.insert(0, 'enviroments')

import enviroment_choose

def run_agent(env, n_games, n_episodes, epsilon=0.01):

    global enviroment_class
    enviroment_class = enviroment_choose.env_choose(env)
    return policy_iterator(env, n_games, n_episodes, epsilon)

def policy_iterator(env, n_games, n_episodes, epsilon=0.01):
    tests_result = []
    policy = create_random_policy(env)
    random_agent_info = {
        "policy": policy,
        "state_action_table": create_state_action_dictionary(env, policy),
        "returns": {}
    }
    random_policy_score = test_policy(policy, env)
    best_agent_info = (random_agent_info, random_policy_score)

    for i in tqdm(range(n_games)):
        new_agent_info =  monte_carlo_control_on_policy(
            env,
            policy=best_agent_info[0]["policy"],
            state_action_table=best_agent_info[0]["state_action_table"],
            returns=best_agent_info[0]["returns"],
            episodes=n_episodes,
            epsilon=epsilon
        )
        new_policy_score = test_policy(new_agent_info["policy"], env)
        tests_result.append(new_policy_score)
        if new_policy_score > best_agent_info[1]:
            best_agent_info = (new_agent_info, new_policy_score)

    dict = {"agent_info": best_agent_info[0], "tests_result": tests_result}
    return dict

def monte_carlo_control_on_policy(env, episodes=100, policy=None, state_action_table=None, returns=None, epsilon=0.01):
    if not policy:
        policy = create_random_policy(env)  # Create an empty dictionary to store state action values

    if not state_action_table:
        Q = create_state_action_dictionary(env, policy) # Empty dictionary for storing rewards for each state-action pair
    else:
        Q = state_action_table

    if not returns:
        returns = {} # 3.


    for _ in range(episodes): # Looping through episodes

        G = 0 # Store cumulative reward in G (initialized at 0)
        episode = enviroment_class.run_game(env=env, policy=policy, display=False) # Store state, action and value respectively

        # for loop through reversed indices of episode array.
        # The logic behind it being reversed is that the eventual reward would be at the end.
        # So we have to go back from the last timestep to the first one propagating result from the future.

        for i in reversed(range(0, len(episode))):
            s_t, a_t, r_t = episode[i]
            state_action = (s_t, a_t)
            G += r_t # Increment total reward by reward on current timestep

            if not state_action in [(x[0], x[1]) for x in episode[0:i]]: #
                if returns.get(state_action):
                    returns[state_action].append(G)
                else:
                    returns[state_action] = [G]

                Q[s_t][a_t] = sum(returns[state_action]) / len(returns[state_action]) # Average reward across episodes

                Q_list = list(map(lambda x: x[1], Q[s_t].items())) # Finding the action with maximum value
                indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
                max_Q = random.choice(indices)

                A_star = max_Q # 14.

                for a in policy[s_t].items(): # Update action probability for s_t in policy
                    if a[0] == A_star:
                        policy[s_t][a[0]] = 1 - epsilon + (epsilon / abs(sum(policy[s_t].values())))
                    else:
                        policy[s_t][a[0]] = (epsilon / abs(sum(policy[s_t].values())))


    agent_info = {"policy": policy, "state_action_table": Q, "returns": returns}

    return agent_info



def create_random_policy(env):
     policy = {}
     for key in range(0, env.observation_space.n):
          current_end = 0
          p = {}
          for action in range(0, env.action_space.n):
               p[action] = 1 / env.action_space.n
          policy[key] = p
     return policy

def create_state_action_dictionary(env, policy):
    Q = {}
    for key in policy.keys():
         Q[key] = {a: 0.0 for a in range(0, env.action_space.n)}
    return Q

def test_policy(policy, env):
    wins = 0
    r = 1000
    for i in range(r):
        w = enviroment_class.run_game(env, policy, display=False)[-1][-1]
        if w == 1:
            wins += 1

    return wins / r
