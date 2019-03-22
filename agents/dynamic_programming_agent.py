import numpy as np
import copy
import random
import enviroment_choose
from tqdm import tqdm
import sys
sys.path.insert(0, 'enviroments')


def run_agent(env, n_games, n_episodes, epsilon=0.01):

    global enviroment_class
    enviroment_class = enviroment_choose.env_choose(env)
    tmp = policy_iteration(env)
    agent_info = {
        "policy": tmp[0],
        "state_action_table": tmp[1]
    }

    policy_dict = policy_matrix_to_dict(agent_info["policy"])

    tests_result = []
    for _ in tqdm(range(0, 100)):

        test = enviroment_class.test_policy(policy_dict, env)
        tests_result.append(test)

    return {"agent_info": agent_info, "tests_result": tests_result}



def policy_iteration(env, gamma=1, theta=1e-8, max_iteration=1e6):
    policy = np.ones([len(enviroment_class.number_states(env)), enviroment_class.number_actions(env)]) / enviroment_class.number_actions(env)
    for _ in tqdm(range(int(max_iteration))):
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V)

        # OPTION 1: stop if the policy is unchanged after an improvement step
        if (new_policy == policy).all():
            break;

        # OPTION 2: stop if the value function estimates for successive policies has converged
        # if np.max(abs(policy_evaluation(env, policy) - policy_evaluation(env, new_policy))) < theta*1e2:
        #    break;

        policy = copy.copy(new_policy)
    return policy, V


def policy_evaluation(env, policy, gamma=1, theta=1e-8):
    V = np.zeros(len(enviroment_class.number_states(env)))
    while True:
        delta = 0
        for s in range(len(enviroment_class.number_states(env))):
            Vs = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in enviroment_class.probability(env)[s][a]: #####################################
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
        for prob, next_state, reward, done in enviroment_class.probability(env)[s][a]: ##############################################à
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
