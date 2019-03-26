import numpy as np
import random
import enviroment_choose
from tqdm import tqdm
import sys
sys.path.insert(0, 'enviroments')


def run_agent(env, alpha=0.1, gamma=0.6, epsilon=0.1, n_games=100, n_episodes=100):

    global enviroment_class
    enviroment_class = enviroment_choose.env_choose(env)
    results = q_learning(env, alpha, gamma, epsilon, n_games, n_episodes)
    tests_result = {}

    for type_test in enviroment_class.type_test():
        tests_result.update({type_test: []})

    for type_test in tests_result:
        for test in results["tests_result"]:
            tests_result[type_test].append(test[type_test])

    return {"agent_info": results["agent_info"], "tests_result": tests_result}



def q_learning(env, alpha=0.1, gamma=0.6, epsilon=0.1, n_games=100, n_episodes=100):

    q_table = np.zeros([len(enviroment_class.number_states(env)), enviroment_class.number_actions(env)])
    tests_result = []

    #Ottengo dall'ambiente i tipi di test che mi può restituire
    type_test_list = enviroment_class.type_test()

    for _ in tqdm(range(0, n_games)):

        for _ in range(0, n_episodes):

            state = env.reset()
            action = 0
            reward = 0
            done = False

            '''
            TRAINING
            '''
            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample() # Explore action space
                else:
                    action = np.argmax(q_table[state]) # Exploit learned values

                next_state, reward, done, info = enviroment_class.run_game(env, action)

                q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
                state = next_state


        '''
        TESTING
        '''
        n_test = 100
        test_iteration_i = {}
        for type_test in type_test_list:
            test_iteration_i.update({type_test: 0})

        for _ in range(n_test):

            done = False
            state = env.reset()

            while not done:
                action = np.argmax(q_table[state]) # Use the best learned action
                test_dict = enviroment_class.test_policy(env, action)
                state = test_dict["env_info"]["next_state"]
                done = test_dict["env_info"]["done"]

                for type_test in type_test_list:
                    test_iteration_i[type_test] += test_dict[type_test]

        for type_test in type_test_list:
            test_iteration_i[type_test] = test_iteration_i[type_test] / n_test

        tests_result.append(test_iteration_i)

    agent_info = {"q_table": q_table}
    return {"agent_info": agent_info, "tests_result": tests_result}
