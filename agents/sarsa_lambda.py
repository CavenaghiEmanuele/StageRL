import sys
import random
import itertools
import numpy as np
from agents.tiles import *
import enviroment_choose
from tqdm import tqdm
sys.path.insert(0, 'enviroments')


def run_agent(env, tests_moment, n_games, n_episodes, alpha=0.1, gamma=0.6, epsilon=0.1, n_step=10, lambd=0.92):

    global _ENVIROMENT_CLASS
    global _ENV
    global _N_GAMES
    global _N_EPISODES
    global _ALPHA
    global _GAMMA
    global _EPSILON
    global _LAMBDA
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
    _LAMBDA = lambd
    _N_STEP = n_step
    _ESTIMATOR = QEstimator(env=_ENV, step_size=_ALPHA, \
        num_tilings=_ENVIROMENT_CLASS.num_tilings(), \
        max_size=_ENVIROMENT_CLASS.IHT_max_size(), trace=True)
    _TESTS_MOMENT = tests_moment


    results = sarsa_lambda()

    tests_result_dict = {}

    for type_test in _TYPE_TEST_LIST:
        tests_result_dict.update({type_test: []})

    for type_test in tests_result_dict:
        for test in results["tests_result"]:
            tests_result_dict[type_test].append(test[type_test])

    return {"agent_info": results["agent_info"], "tests_result": tests_result_dict}


def sarsa_lambda():

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


    agent_info = {"policy": _POLICY}
    return {"agent_info": agent_info, "tests_result": _TESTS_RESULT}


def training():

    # Reset the eligibility trace
    _ESTIMATOR.reset(z_only=True)

    # Reset the environment and pick the first action
    state = _ENVIROMENT_CLASS.reset_env_approximate(_ENV)

    # Take next action
    action_probs = _POLICY(state)
    n = random.uniform(0, sum(action_probs))
    top_range = 0
    action_name = -1
    for prob in action_probs:
        action_name += 1
        top_range += prob
        if n < top_range:
            action = action_name
            break

    # Step through episode
    T = float('inf')
    for t in itertools.count():

        # Take a step
        next_state, reward, done, _ = _ENVIROMENT_CLASS.run_game_approximate(_ENV, action)
        target = reward
        next_action = 0

        if done:
            _ESTIMATOR.update(state, action, target)
            break

        else:
            # Take next action
            action_probs = _POLICY(state)
            n = random.uniform(0, sum(action_probs))
            top_range = 0
            action_name = -1
            for prob in action_probs:
                action_name += 1
                top_range += prob
                if n < top_range:
                    next_action = action_name
                    break

            # Estimate q-value at next state-action
            q_new = _ESTIMATOR.predict(next_state, next_action)[0]
            target = target + _GAMMA * q_new
            # Update step
            _ESTIMATOR.update(state, action, target)
            _ESTIMATOR.z *= _GAMMA * _LAMBDA


        state = next_state
        action = next_action


def testing():

    n_test = 100
    test_iteration_i = {}
    for type_test in _TYPE_TEST_LIST:
        test_iteration_i.update({type_test: 0})

    for _ in range(n_test):


        state = _ENVIROMENT_CLASS.reset_env_approximate(_ENV)
        done = False


        while not done:

            '''
            Scegliere sempre e solo l'azione migliore puo' portare l'agente a restare
            bloccato, con una scelta randomica paghiamo in % di vittorie ma
            evitiamo il problema
            '''
            # Take next action
            action_probs = _POLICY(state)
            n = random.uniform(0, sum(action_probs))
            top_range = 0
            action_name = -1
            for prob in action_probs:
                action_name += 1
                top_range += prob
                if n < top_range:
                    action = action_name
                    break

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


class QEstimator():
    """
    Linear action-value (q-value) function approximator for
    semi-gradient methods with state-action featurization via tile coding.
    """

    def __init__(self, step_size, env, num_tilings=8, max_size=4096, trace=False):

        self.env = env
        self.trace = trace
        self.max_size = max_size
        self.num_tilings = num_tilings
        self.tiling_dim = num_tilings

        # Step size is interpreted as the fraction of the way we want
        # to move towards the target. To compute the learning rate alpha,
        # scale by number of tilings.
        self.alpha = step_size / num_tilings

        # Initialize index hash table (IHT) for tile coding.
        # This assigns a unique index to each tile up to max_size tiles.
        # Ensure max_size >= total number of tiles (num_tilings x tiling_dim x tiling_dim)
        # to ensure no duplicates.
        self.iht = IHT(max_size)

        # Initialize weights (and optional trace)
        self.weights = np.zeros(max_size)
        if self.trace:
            self.z = np.zeros(max_size)

        # Tilecoding software partitions at integer boundaries
        self.features_vector = _ENVIROMENT_CLASS.features_vector(_ENV, self.tiling_dim)

    def featurize_state_action(self, state, action):
        """
        Returns the featurized representation for a
        state-action pair.
        """
        features = []
        for i in range(len(self.features_vector)):
            features.append(self.features_vector[i]*state[i])

        featurized = tiles(self.iht, self.num_tilings, features, [action])
        return featurized

    def predict(self, s, a=None):
        """
        Predicts q-value(s) using linear FA.
        If action a is given then returns prediction
        for single state-action pair (s, a).
        Otherwise returns predictions for all actions
        in environment paired with s.
        """

        if a is None:
            features = [self.featurize_state_action(s, i) for
                        i in range(self.env.action_space.n)]
        else:
            features = [self.featurize_state_action(s, a)]

        return [np.sum(self.weights[f]) for f in features]

    def update(self, s, a, target):
        """
        Updates the estimator parameters
        for a given state and action towards
        the target using the gradient update rule
        (and the eligibility trace if one has been set).
        """
        features = self.featurize_state_action(s, a)
        estimation = np.sum(self.weights[features])  # Linear FA
        delta = (target - estimation)

        if self.trace:
            # self.z[features] += 1  # Accumulating trace
            self.z[features] = 1  # Replacing trace
            self.weights += self.alpha * delta * self.z
        else:
            self.weights[features] += self.alpha * delta

    def reset(self, z_only=False):
        """
        Resets the eligibility trace (must be done at
        the start of every epoch) and optionally the
        weight vector (if we want to restart training
        from scratch).
        """

        if z_only:
            assert self.trace, 'q-value estimator has no z to reset.'
            self.z = np.zeros(self.max_size)
        else:
            if self.trace:
                self.z = np.zeros(self.max_size)
            self.weights = np.zeros(self.max_size)
