import sys
import copy
import numpy as np
import enviroment_choose
from tqdm import tqdm
sys.path.insert(0, 'enviroments')


def run_agent(env, tests_moment, gamma=1, theta=1e-8):

    global _ENVIROMENT_CLASS
    _ENVIROMENT_CLASS = enviroment_choose.env_choose(env)
    tmp = policy_iteration(env, gamma=gamma, theta=theta)
    agent_info = {
        "policy": tmp[0],
        "state_action_table": tmp[1]
    }


    '''
    TESTING
    '''
    #Ottengo dall'ambiente i tipi di test che mi puo' restituire
    type_test_list = _ENVIROMENT_CLASS.type_test()
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
            state = _ENVIROMENT_CLASS.reset_env(env)

            while not done:
                action = np.argmax(agent_info["policy"][state]) # Use the best learned action
                test_dict = _ENVIROMENT_CLASS.test_policy(env, action)
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
    policy = np.ones([len(_ENVIROMENT_CLASS.number_states(env)), \
            _ENVIROMENT_CLASS.number_actions(env)]) / \
            _ENVIROMENT_CLASS.number_actions(env)

    for _ in tqdm(range(int(max_iteration))):
        v = policy_evaluation(env, policy, gamma=gamma, theta=theta)
        new_policy = policy_improvement(env, v, gamma=gamma)

        # Stop if the value function estimates for successive policies has converged
        if np.max(abs(policy_evaluation(env, policy, gamma=gamma, theta=theta) \
            - policy_evaluation(env, new_policy, gamma=gamma, theta=theta))) < theta:
            break

        policy = copy.copy(new_policy)

    return policy, v

def policy_evaluation(env, policy, gamma=1, theta=1e-8):
    v = np.zeros(len(_ENVIROMENT_CLASS.number_states(env)))
    # Tronchiamo la valutazione della policy dopo 500 iterazioni
    # seguendo l'idea di truncated policy
    for _ in range(0, 500):
        delta = 0
        for s in range(len(_ENVIROMENT_CLASS.number_states(env))):
            vs = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, _ in _ENVIROMENT_CLASS.probability(env)[s][a]:
                    vs += action_prob * prob * (reward + gamma * v[next_state])
            delta = max(delta, np.abs(v[s]-vs))
            v[s] = vs
        if delta < theta:
            break
    return v

def policy_improvement(env, v, gamma=1):
    policy = np.zeros([len(_ENVIROMENT_CLASS.number_states(env)), \
        _ENVIROMENT_CLASS.number_actions(env)]) / \
        _ENVIROMENT_CLASS.number_actions(env)
    for s in range(len(_ENVIROMENT_CLASS.number_states(env))):
        q = q_from_v(env, v, s, gamma)

        best_a = np.argwhere(q == np.max(q)).flatten()
        policy[s] = np.sum([np.eye(_ENVIROMENT_CLASS.number_actions(env))[i] \
            for i in best_a], axis=0)/len(best_a)

    return policy

def q_from_v(env, v, s, gamma=1):
    q = np.zeros(_ENVIROMENT_CLASS.number_actions(env))
    for a in range(_ENVIROMENT_CLASS.number_actions(env)):
        for prob, next_state, reward, _ in _ENVIROMENT_CLASS.probability(env)[s][a]:
            q[a] += prob * (reward + gamma * v[next_state])
    return q
