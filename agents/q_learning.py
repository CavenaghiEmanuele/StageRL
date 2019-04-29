import sys
import random
import numpy as np
import enviroment_choose
from tqdm import tqdm
sys.path.insert(0, 'enviroments')


def run_agent(env, tests_moment, n_games, n_episodes, alpha=0.1, gamma=0.6, epsilon=0.1):

    global _ENVIROMENT_CLASS
    global _ENV
    global _N_GAMES
    global _N_EPISODES
    global _ALPHA
    global _GAMMA
    global _EPSILON
    global _TESTS_MOMENT


    _ENVIROMENT_CLASS = enviroment_choose.env_choose(env)
    _ENV = env
    _N_GAMES = n_games
    _N_EPISODES = n_episodes
    _ALPHA = alpha
    _GAMMA = gamma
    _EPSILON = epsilon
    _TESTS_MOMENT = tests_moment

    results = q_learning()

    tests_result_dict = {}

    for type_test in _TYPE_TEST_LIST:
        tests_result_dict.update({type_test: []})

    for type_test in tests_result_dict:
        for test in results["tests_result"]:
            tests_result_dict[type_test].append(test[type_test])

    return {"agent_info": results["agent_info"], "tests_result": tests_result_dict}

def q_learning():

    global _Q_TABLE
    global _TESTS_RESULT
    global _TYPE_TEST_LIST

    _Q_TABLE = np.zeros([len(_ENVIROMENT_CLASS.number_states(_ENV)), \
            _ENVIROMENT_CLASS.number_actions(_ENV)])


    _TESTS_RESULT = []
    #Ottengo dall'ambiente i tipi di test che mi puo' restituire
    _TYPE_TEST_LIST = _ENVIROMENT_CLASS.type_test()


    '''
    TRAINING
    '''
    for i_game in tqdm(range(_N_GAMES)):
        for _ in range(_N_EPISODES):
            training()

        if (i_game % 10) == 0 and _TESTS_MOMENT == "ten_perc":
            testing()

        if _TESTS_MOMENT == "on_run":
            testing()


    '''
    TESTING if type_test is final
    '''
    if _TESTS_MOMENT == "final":
        for _ in range(100):
            testing()



    agent_info = {"q_table": _Q_TABLE}
    return {"agent_info": agent_info, "tests_result": _TESTS_RESULT}

def training():

    state = _ENVIROMENT_CLASS.reset_env(_ENV)
    action = 0
    reward = 0
    done = False

    while not done:

        if random.uniform(0, 1) < _EPSILON:
            action = _ENV.action_space.sample() # Explore action space
        else:
            action = np.argmax(_Q_TABLE[state]) # Exploit learned values

        next_state, reward, done, _ = _ENVIROMENT_CLASS.run_game(_ENV, action)

        _Q_TABLE[state, action] += _ALPHA * \
            (reward + _GAMMA * np.max(_Q_TABLE[next_state]) - _Q_TABLE[state, action])
        state = next_state

def testing():

    n_test = 100
    test_iteration_i = {}
    for type_test in _TYPE_TEST_LIST:
        test_iteration_i.update({type_test: 0})

    for _ in range(n_test):

        done = False
        state = _ENVIROMENT_CLASS.reset_env(_ENV)

        while not done:

            if random.uniform(0, 1) < _EPSILON:
                action = _ENV.action_space.sample() # Explore action space
            else:
                action = np.argmax(_Q_TABLE[state]) # Exploit learned values

            test_dict = _ENVIROMENT_CLASS.test_policy(_ENV, action)
            state = test_dict["env_info"]["next_state"]
            done = test_dict["env_info"]["done"]

            for type_test in _TYPE_TEST_LIST:
                test_iteration_i[type_test] += test_dict[type_test]

    for type_test in _TYPE_TEST_LIST:
        test_iteration_i[type_test] = test_iteration_i[type_test] / n_test

    _TESTS_RESULT.append(test_iteration_i)
