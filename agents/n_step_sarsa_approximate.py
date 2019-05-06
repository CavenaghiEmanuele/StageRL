import sys
import itertools
import numpy as np
import enviroment_choose
from tqdm import tqdm
sys.path.insert(0, 'enviroments')


def run_agent(env, tests_moment, n_games, n_episodes, alpha=0.1, gamma=0.6, epsilon=0.1, n_step=10):

    global _ENVIROMENT_CLASS
    global _ENV
    global _N_GAMES
    global _N_EPISODES
    global _ALPHA
    global _GAMMA
    global _EPSILON
    global _N_STEP
    global _ESTIMATOR
    global _TESTS_MOMENT


    _ENVIROMENT_CLASS = enviroment_choose.env_choose(env)
    _ENV = env
    _N_GAMES = n_games
    _N_EPISODES = n_episodes
    _ALPHA = alpha
    _GAMMA = gamma
    _EPSILON = epsilon
    _N_STEP = n_step
    _ESTIMATOR = _ENVIROMENT_CLASS.QEstimator(env=_ENV, step_size=_ALPHA)
    _TESTS_MOMENT = tests_moment


    results = n_step_sarsa_approximate()

    tests_result_dict = {}

    for type_test in _TYPE_TEST_LIST:
        tests_result_dict.update({type_test: []})

    for type_test in tests_result_dict:
        for test in results["tests_result"]:
            tests_result_dict[type_test].append(test[type_test])

    return {"agent_info": results["agent_info"], "tests_result": tests_result_dict}


def n_step_sarsa_approximate():

    global _POLICY
    global _TYPE_TEST_LIST
    global _TESTS_RESULT

    _TESTS_RESULT = []
    #Ottengo dall'ambiente i tipi di test che mi puo' restituire
    _TYPE_TEST_LIST = _ENVIROMENT_CLASS.type_test()
    # Create epsilon-greedy policy
    _POLICY = make_epsilon_greedy_policy()

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

    print(_POLICY)
    agent_info = {"policy": _POLICY}
    return {"agent_info": agent_info, "tests_result": _TESTS_RESULT}


def training():

    # Reset the environment and pick the first action
    state = _ENVIROMENT_CLASS.reset_env_approximate(_ENV)

    # Take next action
    action_probs = _POLICY(state)
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

    # Set up trackers
    states = [state]
    actions = [action]
    rewards = [0.0]

    # Step through episode
    T = float('inf')
    for t in itertools.count():
        if t < T:
            # Take a step
            next_state, reward, done, _ = _ENVIROMENT_CLASS.run_game_approximate(_ENV, action)
            states.append(next_state)
            rewards.append(reward)
            next_action = 0

            if done:
                T = t + 1

            else:
                # Take next step
                next_action_probs = _POLICY(next_state)
                next_action = np.random.choice(
                    np.arange(len(next_action_probs)), p=next_action_probs)

                actions.append(next_action)

        update_time = t + 1 - _N_STEP  # Specifies state to be updated
        if update_time >= 0:
            # Build target
            g = 0
            for i in range(update_time + 1, min(T, update_time + _N_STEP) + 1):
                g += np.power(_GAMMA, i - update_time - 1) * rewards[i]
            if update_time + _N_STEP < T:
                q_values_next = _ESTIMATOR.predict(states[update_time + _N_STEP])
                g += q_values_next[actions[update_time + _N_STEP]]

            # Update policy
            _ESTIMATOR.update(states[update_time], actions[update_time], g)

        if update_time == T - 1:
            break

        state = next_state
        action = next_action


def testing():

    n_test = 10
    test_iteration_i = {}
    for type_test in _TYPE_TEST_LIST:
        test_iteration_i.update({type_test: 0})

    for _ in range(n_test):


        state = _ENVIROMENT_CLASS.reset_env_approximate(_ENV)
        done = False


        while not done:

            # Take next action
            action_probs = _POLICY(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            '''
            Scegliere sempre e solo l'azione migliore puo' portare l'agente a restare
            bloccato, con una scelta randomica paghiamo in % di vittorie ma
            evitiamo il problema
            '''
            test_dict = _ENVIROMENT_CLASS.test_policy_approximate(_ENV, action)
            state = test_dict["env_info"]["next_state"]
            done = test_dict["env_info"]["done"]

            for type_test in _TYPE_TEST_LIST:
                test_iteration_i[type_test] += test_dict[type_test]

    for type_test in _TYPE_TEST_LIST:
        test_iteration_i[type_test] = test_iteration_i[type_test] / n_test

    _TESTS_RESULT.append(test_iteration_i)


def make_epsilon_greedy_policy():
    """
    Creates an epsilon-greedy policy based on a
    given q-value approximator and epsilon.
    """
    def policy_fn(observation):
        action_probs = np.ones(_ENVIROMENT_CLASS.number_actions(_ENV), dtype=float) \
            * _EPSILON / _ENVIROMENT_CLASS.number_actions(_ENV)
        q_values = _ESTIMATOR.predict(observation)
        best_action_idx = np.argmax(q_values)
        action_probs[best_action_idx] += (1.0 - _EPSILON)
        return action_probs
    return policy_fn
