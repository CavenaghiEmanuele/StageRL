import numpy as np
import random
import enviroment_choose
from tqdm import tqdm
import sys
sys.path.insert(0, 'enviroments')


def run_agent(env, alpha=0.1, gamma=0.6, epsilon=0.1, n_games=100, n_episodes=100):

    global enviroment_class
    enviroment_class = enviroment_choose.env_choose(env)
    return q_learning(env, alpha, gamma, epsilon, n_games, n_episodes)



def q_learning(env, alpha=0.1, gamma=0.6, epsilon=0.1, n_games=100, n_episodes=100):

    q_table = np.zeros([len(enviroment_class.number_states(env)), enviroment_class.number_actions(env)])
    tests_result = []

    for _ in tqdm(range(0, n_games)):

        for _ in range(0, n_episodes):

            state = env.reset()
            epochs, penalties, reward, = 0, 0, 0
            done = False

            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample() # Explore action space
                else:
                    action = np.argmax(q_table[state]) # Exploit learned values

                # Non viene usato Run_game in quanto non esiste una policy ma solo la q_table
                next_state, reward, done, info = env.step(action)

                q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

                state = next_state


        dict_q_table = policy_matrix_to_dict(q_table)
        tests_result.append(enviroment_class.test_policy(dict_q_table, env))

    print(q_table)
    agent_info = {"q_table": q_table}
    return {"agent_info": agent_info, "tests_result": tests_result}


def policy_matrix_to_dict(policy):

    dict = {}
    for state in range(len(policy)):
        p = {}
        for action in range(len(policy[state])):
            p[action] = policy[state][action]
        dict[state] = p
    return dict
