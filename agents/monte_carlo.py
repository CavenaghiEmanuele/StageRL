import numpy as np
import random
from tqdm import tqdm
import sys
sys.path.insert(0, 'enviroments')
import enviroment_choose


def run_agent(env, n_games, n_episodes, epsilon=0.01):

    global enviroment_class
    enviroment_class = enviroment_choose.env_choose(env)
    results = monte_carlo_control(env, n_games, n_episodes, epsilon)

    tests_result = {}

    for type_test in enviroment_class.type_test():
        tests_result.update({type_test: []})

    for type_test in tests_result:
        for test in results["tests_result"]:
            tests_result[type_test].append(test[type_test])

    return {"agent_info": results["agent_info"], "tests_result": tests_result}





def monte_carlo_control(env, n_games, n_episodes, epsilon):

    policy = np.ones([len(enviroment_class.number_states(env)), enviroment_class.number_actions(env)]) / enviroment_class.number_actions(env)
    Q = np.zeros([len(enviroment_class.number_states(env)), enviroment_class.number_actions(env)])
    returns_number = {}

    tests_result = []
    #Ottengo dall'ambiente i tipi di test che mi può restituire
    type_test_list = enviroment_class.type_test()

    for _ in tqdm(range(n_games)):

        for _ in range(n_episodes): 

            G = 0 # Store cumulative reward in G (initialized at 0)
            episode = []

            state = enviroment_class.reset_env(env)
            action = 0
            reward = 0
            done = False

            '''
            TRAINING
            '''
            while not done:

                n = random.uniform(0, sum(policy[state]))
                top_range = 0
                action_name = -1
                for prob in policy[state]:
                    action_name += 1
                    top_range += prob
                    if n < top_range:
                        action = action_name
                        break
                '''
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample() # Explore action space
                else:
                    action = np.argmax(policy[state]) # Exploit learned values
                '''
                next_state, reward, done, info = enviroment_class.run_game(env, action)
                episode.append([state, action, reward])
                state = next_state



            for i in reversed(range(0, len(episode))):
                s_t, a_t, r_t = episode[i]
                state_action = (s_t, a_t)
                G += r_t # Increment total reward by reward on current timestep
                if not state_action in [(x[0], x[1]) for x in episode[0:i]]: #because is first visit algorithm

                    if returns_number.get(state_action):
                        returns_number[state_action] += 1
                        Q[s_t][a_t] = Q[s_t][a_t] + ((1 / returns_number[state_action]) * (G - Q[s_t][a_t]))
                    else:
                        returns_number[state_action] = 1
                        Q[s_t][a_t] = G

                    # Finding the action with maximum value
                    indices = [i for i, x in enumerate(Q[s_t]) if x == max(Q[s_t])]
                    max_Q = random.choice(indices)

                    A_star = max_Q

                    for a in range(len(policy[s_t])): # Update action probability for s_t in policy
                        if a == A_star:
                            policy[s_t][a] = 1 - epsilon + (epsilon / abs(sum(policy[s_t])))
                        else:
                            policy[s_t][a] = (epsilon / abs(sum(policy[s_t])))

        '''
        TESTING
        '''
        n_test = 100
        test_iteration_i = {}
        for type_test in type_test_list:
            test_iteration_i.update({type_test: 0})

        for _ in range(n_test):

            done = False
            state = enviroment_class.reset_env(env)

            while not done:

                n = random.uniform(0, sum(policy[state]))
                top_range = 0
                action_name = -1
                for prob in policy[state]:
                    action_name += 1
                    top_range += prob
                    if n < top_range:
                        action = action_name
                        break
                '''
                Scegliere sempre e solo l'azione migliore può portare l'agente a restare
                bloccato, con una scelta randomica paghiamo in % di vittorie ma
                evitiamo il problema
                action = np.argmax(policy[state]) # Use the best learned action
                '''
                test_dict = enviroment_class.test_policy(env, action)
                state = test_dict["env_info"]["next_state"]
                done = test_dict["env_info"]["done"]

                for type_test in type_test_list:
                    test_iteration_i[type_test] += test_dict[type_test]

        for type_test in type_test_list:
            test_iteration_i[type_test] = test_iteration_i[type_test] / n_test

        tests_result.append(test_iteration_i)

    agent_info = {"policy": policy, "state_action_table": Q, "returns_number": returns_number}
    return {"agent_info": agent_info, "tests_result": tests_result}
