import sys
import random
import numpy as np
from tqdm import tqdm
sys.path.insert(0, 'enviroments')
import enviroment_choose




def run_agent(env, tests_moment, n_games, n_episodes, epsilon=0.01, gamma=1):

    global _ENVIROMENT_CLASS
    global _ENV
    global _N_GAMES
    global _N_EPISODES
    global _EPSILON
    global _GAMMA
    global _TESTS_MOMENT

    _ENVIROMENT_CLASS = enviroment_choose.env_choose(env)
    _ENV = env
    _N_GAMES = n_games
    _N_EPISODES = n_episodes
    _EPSILON = epsilon
    _GAMMA = gamma
    _TESTS_MOMENT = tests_moment

    results = monte_carlo_control()


    tests_result_dict = {}

    for type_test in _TYPE_TEST_LIST:
        tests_result_dict.update({type_test: []})

    for type_test in tests_result_dict:
        for test in results["tests_result"]:
            tests_result_dict[type_test].append(test[type_test])


    return {"agent_info": results["agent_info"], "tests_result": tests_result_dict}


def monte_carlo_control():

    global _POLICY
    global _Q
    global _RETURNS_NUMBER
    global _TESTS_RESULT
    global _TYPE_TEST_LIST

    _POLICY = np.ones([len(_ENVIROMENT_CLASS.number_states(_ENV)), \
            _ENVIROMENT_CLASS.number_actions(_ENV)]) / \
            _ENVIROMENT_CLASS.number_actions(_ENV)
    _Q = np.zeros([len(_ENVIROMENT_CLASS.number_states(_ENV)), \
            _ENVIROMENT_CLASS.number_actions(_ENV)])
    _RETURNS_NUMBER = {}


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

    agent_info = {"policy": _POLICY, "state_action_table": _Q, "returns_number": _RETURNS_NUMBER}
    return {"agent_info": agent_info, "tests_result": _TESTS_RESULT}


def training():

    g = 0 # Store cumulative reward in G (initialized at 0)
    episode = []

    state = _ENVIROMENT_CLASS.reset_env(_ENV)
    action = 0
    reward = 0
    done = False

    while not done:

        n = random.uniform(0, sum(_POLICY[state]))
        top_range = 0
        action_name = -1
        for prob in _POLICY[state]:
            action_name += 1
            top_range += prob
            if n < top_range:
                action = action_name
                break


        next_state, reward, done, _ = _ENVIROMENT_CLASS.run_game(_ENV, action)
        episode.append([state, action, reward])
        state = next_state


    for i in reversed(range(0, len(episode))):
        s_t, a_t, r_t = episode[i]
        state_action = (s_t, a_t)
        # Increment total reward by reward on current timestep
        g = (_GAMMA * g) + r_t

        #because is first visit algorithm
        if not state_action in [(x[0], x[1]) for x in episode[0:i]]:

            if _RETURNS_NUMBER.get(state_action):
                _RETURNS_NUMBER[state_action] += 1
                #Incremental implementation
                _Q[s_t][a_t] = _Q[s_t][a_t] + \
                    ((1 / _RETURNS_NUMBER[state_action]) * (g - _Q[s_t][a_t]))
            else:
                _RETURNS_NUMBER[state_action] = 1
                _Q[s_t][a_t] = g

            # Finding the action with maximum value
            indices = [i for i, x in enumerate(_Q[s_t]) if x == max(_Q[s_t])]
            max_q = random.choice(indices)

            a_star = max_q

            for a in range(len(_POLICY[s_t])): # Update action probability for s_t in policy
                if a == a_star:
                    _POLICY[s_t][a] = 1 - _EPSILON + (_EPSILON / abs(sum(_POLICY[s_t])))
                else:
                    _POLICY[s_t][a] = (_EPSILON / abs(sum(_POLICY[s_t])))


def testing():

    '''
    TESTING
    '''
    n_test = 100
    test_iteration_i = {}

    for type_test in _TYPE_TEST_LIST:
        test_iteration_i.update({type_test: 0})

    for _ in range(n_test):

        done = False
        state = _ENVIROMENT_CLASS.reset_env(_ENV)

        while not done:

            n = random.uniform(0, sum(_POLICY[state]))
            top_range = 0
            action_name = -1
            for prob in _POLICY[state]:
                action_name += 1
                top_range += prob
                if n < top_range:
                    action = action_name
                    break
            '''
            Scegliere sempre e solo l'azione migliore puo' portare l'agente a restare
            bloccato, con una scelta randomica paghiamo in % di vittorie ma
            evitiamo il problema
            '''
            test_dict = _ENVIROMENT_CLASS.test_policy(_ENV, action)
            state = test_dict["env_info"]["next_state"]
            done = test_dict["env_info"]["done"]

            for type_test in _TYPE_TEST_LIST:
                test_iteration_i[type_test] += test_dict[type_test]

    for type_test in _TYPE_TEST_LIST:
        test_iteration_i[type_test] = test_iteration_i[type_test] / n_test

    _TESTS_RESULT.append(test_iteration_i)
