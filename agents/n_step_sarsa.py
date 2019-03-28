import numpy as np
import random
import enviroment_choose
import itertools
from tqdm import tqdm
import sys
sys.path.insert(0, 'enviroments')


def run_agent(env, n_games=100, n_episodes=100, alpha=0.1, gamma=0.6, epsilon=0.1, n_step=10):

    global enviroment_class
    enviroment_class = enviroment_choose.env_choose(env)
    results = n_step_sarsa(env, n_games, n_episodes, alpha, gamma, epsilon, n_step)
    tests_result = {}

    for type_test in enviroment_class.type_test():
        tests_result.update({type_test: []})

    for type_test in tests_result:
        for test in results["tests_result"]:
            tests_result[type_test].append(test[type_test])

    return {"agent_info": results["agent_info"], "tests_result": tests_result}




def n_step_sarsa(env, n_games, n_episodes, alpha, gamma, epsilon, n_step):

    """
    n-step semi-gradient Sarsa algorithm
    for finding optimal q and pi via Linear
    FA with n-step TD updates.
    """

    policy = np.ones([len(enviroment_class.number_states(env)), enviroment_class.number_actions(env)]) / enviroment_class.number_actions(env)
    Q = np.zeros([len(enviroment_class.number_states(env)), enviroment_class.number_actions(env)])


    tests_result = []
    #Ottengo dall'ambiente i tipi di test che mi può restituire
    type_test_list = enviroment_class.type_test()


    for _ in tqdm(range(n_games)):

        for _ in range(n_episodes):

            '''
            TRAINING
            '''
            state = enviroment_class.reset_env(env)# Reset the environment and pick the first action

            # Take next step
            n = random.uniform(0, sum(policy[state]))
            top_range = 0
            action_name = -1
            for prob in policy[state]:
                action_name += 1
                top_range += prob
                if n < top_range:
                    action = action_name
                    break


            states = [state]
            actions = [action]
            rewards = [0.0]

            # Step through episode
            T = float('inf')
            for t in itertools.count():
                if t < T:
                    # Take a step
                    next_state, reward, done, _ = enviroment_class.run_game(env, action)
                    states.append(next_state)
                    rewards.append(reward)

                    if done:
                        T = t + 1

                    else:
                        # Take next step
                        n = random.uniform(0, sum(policy[next_state]))
                        top_range = 0
                        action_name = -1
                        for prob in policy[state]:
                            action_name += 1
                            top_range += prob
                            if n < top_range:
                                next_action = action_name
                                break

                        actions.append(next_action)

                update_time = t + 1 - n_step  # Specifies state to be updated
                if update_time >= 0:
                    # Build target
                    G = 0
                    for i in range(update_time + 1, min(T, update_time + n_step) + 1):
                        G += np.power(gamma, i - update_time - 1) * rewards[i]
                    if update_time + n_step < T:
                        G += np.power(gamma,n_step)*Q[states[update_time + n_step]][actions[update_time + n_step]]
                    Q[states[update_time]][actions[update_time]] += alpha * (G - Q[states[update_time]][actions[update_time]])


                        # Finding the action with maximum value
                    indices = [i for i, x in enumerate(Q[next_state]) if x == max(Q[next_state])]
                    max_Q = random.choice(indices)
                    A_star = max_Q

                    for a in range(len(policy[next_state])): # Update action probability for s_t in policy

                        if a == A_star:
                            policy[next_state][a] = 1 - epsilon + (epsilon / abs(sum(policy[next_state])))
                        else:
                            policy[next_state][a] = (epsilon / abs(sum(policy[next_state])))

                if update_time == T - 1:
                    break

                state = next_state
                action = next_action


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

    print(policy)
    print(Q)
    agent_info = {"policy": policy, "state_action_table": Q}
    return {"agent_info": agent_info, "tests_result": tests_result}
