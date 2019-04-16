import numpy as np
import random
from tqdm import tqdm
import sys
sys.path.insert(0, 'enviroments')
import enviroment_choose


def run_agent(env, tests_moment, n_games, n_episodes, epsilon=0.01):

    global _enviroment_class
    global _env
    global _n_games
    global _n_episodes
    global _epsilon
    global _tests_moment

    _enviroment_class = enviroment_choose.env_choose(env)
    _env = env
    _n_games = n_games
    _n_episodes = n_episodes
    _epsilon = epsilon
    _tests_moment = tests_moment

    results = monte_carlo_control()


    tests_result_dict = {}

    for type_test in _type_test_list:
        tests_result_dict.update({type_test: []})

    for type_test in tests_result_dict:
        for test in results["tests_result"]:
            tests_result_dict[type_test].append(test[type_test])


    return {"agent_info": results["agent_info"], "tests_result": tests_result_dict}

def monte_carlo_control():

    global _policy
    global _Q
    global _returns_number
    global _tests_result
    global _type_test_list

    _policy = np.ones([len(_enviroment_class.number_states(_env)), _enviroment_class.number_actions(_env)]) / _enviroment_class.number_actions(_env)
    _Q = np.zeros([len(_enviroment_class.number_states(_env)), _enviroment_class.number_actions(_env)])
    _returns_number = {}


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


    agent_info = {"policy": _policy, "state_action_table": _Q, "returns_number": _returns_number}
    return {"agent_info": agent_info, "tests_result": _tests_result}

def training():

    G = 0 # Store cumulative reward in G (initialized at 0)
    episode = []

    state = _enviroment_class.reset_env(_env)
    action = 0
    reward = 0
    done = False

    while not done:

        n = random.uniform(0, sum(_policy[state]))
        top_range = 0
        action_name = -1
        for prob in _policy[state]:
            action_name += 1
            top_range += prob
            if n < top_range:
                action = action_name
                break


        next_state, reward, done, info = _enviroment_class.run_game(_env, action)
        episode.append([state, action, reward])
        state = next_state


    for i in reversed(range(0, len(episode))):
        s_t, a_t, r_t = episode[i]
        state_action = (s_t, a_t)
        G += r_t # Increment total reward by reward on current timestep
        if not state_action in [(x[0], x[1]) for x in episode[0:i]]: #because is first visit algorithm

            if _returns_number.get(state_action):
                _returns_number[state_action] += 1
                _Q[s_t][a_t] = _Q[s_t][a_t] + ((1 / _returns_number[state_action]) * (G - _Q[s_t][a_t]))
            else:
                _returns_number[state_action] = 1
                _Q[s_t][a_t] = G

            # Finding the action with maximum value
            indices = [i for i, x in enumerate(_Q[s_t]) if x == max(_Q[s_t])]
            max_Q = random.choice(indices)

            A_star = max_Q

            for a in range(len(_policy[s_t])): # Update action probability for s_t in policy
                if a == A_star:
                    _policy[s_t][a] = 1 - _epsilon + (_epsilon / abs(sum(_policy[s_t])))
                else:
                    _policy[s_t][a] = (_epsilon / abs(sum(_policy[s_t])))

def testing():

    '''
    TESTING
    '''
    n_test = 100
    test_iteration_i = {}

    for type_test in _type_test_list:
        test_iteration_i.update({type_test: 0})

    for _ in range(n_test):

        done = False
        state = _enviroment_class.reset_env(_env)

        while not done:

            n = random.uniform(0, sum(_policy[state]))
            top_range = 0
            action_name = -1
            for prob in _policy[state]:
                action_name += 1
                top_range += prob
                if n < top_range:
                    action = action_name
                    break
            '''
            Scegliere sempre e solo l'azione migliore può portare l'agente a restare
            bloccato, con una scelta randomica paghiamo in % di vittorie ma
            evitiamo il problema
            '''
            test_dict = _enviroment_class.test_policy(_env, action)
            state = test_dict["env_info"]["next_state"]
            done = test_dict["env_info"]["done"]

            for type_test in _type_test_list:
                test_iteration_i[type_test] += test_dict[type_test]

    for type_test in _type_test_list:
        test_iteration_i[type_test] = test_iteration_i[type_test] / n_test

    _tests_result.append(test_iteration_i)
