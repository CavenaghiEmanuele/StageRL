import numpy as np
import random
import enviroment_choose
from tqdm import tqdm
import sys
sys.path.insert(0, 'enviroments')


def run_agent(env, tests_moment, n_games, n_episodes, alpha=0.1, gamma=0.6, epsilon=0.1):

    global _enviroment_class
    global _env
    global _n_games
    global _n_episodes
    global _alpha
    global _gamma
    global _epsilon
    global _tests_moment


    _enviroment_class = enviroment_choose.env_choose(env)
    _env = env
    _n_games = n_games
    _n_episodes = n_episodes
    _alpha = alpha
    _gamma = gamma
    _epsilon = epsilon
    _tests_moment = tests_moment

    results = q_learning()

    tests_result_dict = {}

    for type_test in _type_test_list:
        tests_result_dict.update({type_test: []})

    for type_test in tests_result_dict:
        for test in results["tests_result"]:
            tests_result_dict[type_test].append(test[type_test])

    return {"agent_info": results["agent_info"], "tests_result": tests_result_dict}

def q_learning():

    global _q_table
    global _tests_result
    global _type_test_list

    _q_table = np.zeros([len(_enviroment_class.number_states(_env)), _enviroment_class.number_actions(_env)])


    _tests_result = []
    _type_test_list = _enviroment_class.type_test() #Ottengo dall'ambiente i tipi di test che mi può restituire


    '''
    TRAINING
    '''
    for i_game in tqdm(range(_n_games)):
        for _ in range(_n_episodes):
            training()

        if (i_game % 10) == 0 and _tests_moment == "ten_perc":
            testing()

        if _tests_moment == "on_run":
            testing()


    '''
    TESTING if type_test is final
    '''
    if _tests_moment == "final":
        for _ in range(100):
            testing()



    agent_info = {"q_table": _q_table}
    return {"agent_info": agent_info, "tests_result": _tests_result}

def training():

    state = _enviroment_class.reset_env(_env)
    action = 0
    reward = 0
    done = False

    while not done:

        if random.uniform(0, 1) < _epsilon:
            action = _env.action_space.sample() # Explore action space
        else:
            action = np.argmax(_q_table[state]) # Exploit learned values

        next_state, reward, done, info = _enviroment_class.run_game(_env, action)

        _q_table[state, action] += _alpha * (reward + _gamma * np.max(_q_table[next_state]) - _q_table[state, action])
        state = next_state

def testing():

    n_test = 100
    test_iteration_i = {}
    for type_test in _type_test_list:
        test_iteration_i.update({type_test: 0})

    for _ in range(n_test):

        done = False
        state = _enviroment_class.reset_env(_env)

        while not done:

            if random.uniform(0, 1) < _epsilon:
                action = _env.action_space.sample() # Explore action space
            else:
                action = np.argmax(_q_table[state]) # Exploit learned values

            test_dict = _enviroment_class.test_policy(_env, action)
            state = test_dict["env_info"]["next_state"]
            done = test_dict["env_info"]["done"]

            for type_test in _type_test_list:
                test_iteration_i[type_test] += test_dict[type_test]

    for type_test in _type_test_list:
        test_iteration_i[type_test] = test_iteration_i[type_test] / n_test

    _tests_result.append(test_iteration_i)
