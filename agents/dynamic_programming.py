import numpy as np
import copy
import random
import enviroment_choose
from tqdm import tqdm
import sys
sys.path.insert(0, 'enviroments')


def run_agent(env, gamma=1, theta=1e-8, max_iteration=1e6):

    global enviroment_class
    enviroment_class = enviroment_choose.env_choose(env)
    tmp = policy_iteration(env, gamma=gamma, theta=theta)
    agent_info = {
        "policy": tmp[0],
        "state_action_table": tmp[1]
    }


    '''
    TESTING
    '''
    #Ottengo dall'ambiente i tipi di test che mi può restituire
    type_test_list = enviroment_class.type_test()
    tests_result = []
    tmp_tests_result = {}
    n_test = 100
    n_episodes_test = 100

    for type_test in type_test_list:
        tmp_tests_result.update({type_test: []})


    for _ in tqdm(range(n_test)):

        test_iteration_i = {}
        for type_test in type_test_list:
            test_iteration_i.update({type_test: 0})

        #Per ogni test eseguiamo 100 "episodi"
        for _ in range(n_episodes_test):

            done = False
            state = enviroment_class.reset_env(env)

            while not done:
                action = np.argmax(agent_info["policy"][state]) # Use the best learned action
                test_dict = enviroment_class.test_policy(env, action)
                state = test_dict["env_info"]["next_state"]
                done = test_dict["env_info"]["done"]

                for type_test in type_test_list:
                    test_iteration_i[type_test] += test_dict[type_test]

        for type_test in type_test_list:
            test_iteration_i[type_test] = test_iteration_i[type_test] / n_episodes_test

        tests_result.append(test_iteration_i)


    for type_test in tmp_tests_result:
        for test in tests_result:
            tmp_tests_result[type_test].append(test[type_test])


    return {"agent_info": agent_info, "tests_result": tmp_tests_result}



def policy_iteration(env, gamma=1, theta=1e-8, max_iteration=1e6):
    policy = np.ones([len(enviroment_class.number_states(env)), enviroment_class.number_actions(env)]) / enviroment_class.number_actions(env)

    for _ in tqdm(range(int(max_iteration))):
        V = policy_evaluation(env, policy, gamma=gamma, theta=theta)
        new_policy = policy_improvement(env, V, gamma=gamma)

        # Stop if the value function estimates for successive policies has converged
        if np.max(abs(policy_evaluation(env, policy, gamma=gamma, theta=theta) - policy_evaluation(env, new_policy, gamma=gamma, theta=theta))) < theta*1e2:
            break;

        policy = copy.copy(new_policy)

    return policy, V


def policy_evaluation(env, policy, gamma=1, theta=1e-8):
    V = np.zeros(len(enviroment_class.number_states(env)))
    # Tronchiamo la valutazione della policy dopo 500 iterazioni
    # seguendo l'idea di truncated policy
    for i in range(0, 500):
        delta = 0
        for s in range(len(enviroment_class.number_states(env))):
            Vs = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in enviroment_class.probability(env)[s][a]:
                    Vs += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(V[s]-Vs))
            V[s] = Vs
        if delta < theta:
            break
    return V


def policy_improvement(env, V, gamma=1):
    policy = np.zeros([len(enviroment_class.number_states(env)), enviroment_class.number_actions(env)]) / enviroment_class.number_actions(env)
    for s in range(len(enviroment_class.number_states(env))):
        q = q_from_v(env, V, s, gamma)

        best_a = np.argwhere(q==np.max(q)).flatten()
        policy[s] = np.sum([np.eye(enviroment_class.number_actions(env))[i] for i in best_a], axis=0)/len(best_a)

    return policy


def q_from_v(env, V, s, gamma=1):
    q = np.zeros(enviroment_class.number_actions(env))
    for a in range(enviroment_class.number_actions(env)):
        for prob, next_state, reward, done in enviroment_class.probability(env)[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])
    return q


def policy_matrix_to_dict(policy):

    dict = {}
    for state in range(len(policy)):
        p = {}
        for action in range(len(policy[state])):
            p[action] = policy[state][action]
        dict[state] = p
    return dict
