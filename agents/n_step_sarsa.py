import numpy as np
import random
import enviroment_choose
import itertools
from tqdm import tqdm
import sys
sys.path.insert(0, 'enviroments')


def run_agent(env, tests_moment, n_games, n_episodes, alpha=0.1, gamma=0.6, epsilon=0.1, n_step=10):

    global _enviroment_class
    global _env
    global _n_games
    global _n_episodes
    global _alpha
    global _gamma
    global _epsilon
    global _n_step
    global _tests_moment

    _enviroment_class = enviroment_choose.env_choose(env)
    _env = env
    _n_games = n_games
    _n_episodes = n_episodes
    _alpha = alpha
    _gamma = gamma
    _epsilon = epsilon
    _n_step = n_step
    _tests_moment = tests_moment

    results = n_step_sarsa()
    tests_result_dict = {}

    for type_test in _type_test_list:
        tests_result_dict.update({type_test: []})

    for type_test in tests_result_dict:
        for test in results["tests_result"]:
            tests_result_dict[type_test].append(test[type_test])

    return {"agent_info": results["agent_info"], "tests_result": tests_result_dict}

def n_step_sarsa():

    global _policy
    global _Q
    global _type_test_list
    global _tests_result


    _policy = np.ones([len(_enviroment_class.number_states(_env)), _enviroment_class.number_actions(_env)]) / _enviroment_class.number_actions(_env)
    _Q = np.zeros([len(_enviroment_class.number_states(_env)), _enviroment_class.number_actions(_env)])


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



    agent_info = {"policy": _policy, "state_action_table": _Q}
    return {"agent_info": agent_info, "tests_result": _tests_result}

def training():

    state = _enviroment_class.reset_env(_env) # Reset the environment and pick the first action

    # Take next step
    n = random.uniform(0, sum(_policy[state]))
    top_range = 0
    action_name = -1
    for prob in _policy[state]:
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
            next_state, reward, done, _ = _enviroment_class.run_game(_env, action)
            states.append(next_state)
            rewards.append(reward)
            next_action = 0

            if done:
                T = t + 1

            else:
                # Take next step
                n = random.uniform(0, sum(_policy[next_state]))
                top_range = 0
                action_name = -1
                for prob in _policy[state]:
                    action_name += 1
                    top_range += prob
                    if n < top_range:
                        next_action = action_name
                        break

                actions.append(next_action)

        update_time = t + 1 - _n_step  # Specifies state to be updated
        if update_time >= 0:
            # Build target
            G = 0
            for i in range(update_time + 1, min(T, update_time + _n_step) + 1):
                G += np.power(_gamma, i - update_time - 1) * rewards[i]
            if update_time + _n_step < T:
                G += np.power(_gamma, _n_step)*_Q[states[update_time + _n_step]][actions[update_time + _n_step]]
            _Q[states[update_time]][actions[update_time]] += _alpha * (G - _Q[states[update_time]][actions[update_time]])


            # Finding the action with maximum value
            indices = [i for i, x in enumerate(_Q[next_state]) if x == max(_Q[next_state])]
            max_Q = random.choice(indices)
            A_star = max_Q

            for a in range(len(_policy[next_state])): # Update action probability for s_t in policy

                if a == A_star:
                    _policy[next_state][a] = 1 - _epsilon + (_epsilon / abs(sum(_policy[next_state])))
                else:
                    _policy[next_state][a] = (_epsilon / abs(sum(_policy[next_state])))

        if update_time == T - 1:
            break

        state = next_state
        action = next_action

def testing():

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
