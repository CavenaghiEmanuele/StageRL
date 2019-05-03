import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import time
import timeit
from collections import namedtuple
import os
import glob
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib import cm
matplotlib.style.use('ggplot')

import io
import base64
from IPython.display import HTML
from agents.tiles import *
sys.path.insert(0, 'enviroments')

import enviroment_choose



def run_agent(env, n_games, n_episodes, alpha=0.1, gamma=0.6, epsilon=0.1, n_step=10):

    global _ENVIROMENT_CLASS
    global _ENV
    global _N_GAMES
    global _N_EPISODES
    global _ALPHA
    global _GAMMA
    global _EPSILON
    global _N_STEP
    global _ESTIMATOR


    _ENVIROMENT_CLASS = enviroment_choose.env_choose(env)
    _ENV = env
    _N_GAMES = n_games
    _N_EPISODES = n_episodes
    _ALPHA = alpha
    _GAMMA = gamma
    _EPSILON = epsilon
    _N_STEP = n_step
    _ESTIMATOR = _ENVIROMENT_CLASS.QEstimator(env=_ENV, step_size=_ALPHA)




    start_time = timeit.default_timer()
    n_step_sarsa_approximate()
    elapsed_time = timeit.default_timer() - start_time
    print('{} episodes completed in {:.2f}s'.format(_N_GAMES, elapsed_time))





def n_step_sarsa_approximate():

    global _POLICY


    '''
    TRAINING
    '''
    for i_game in tqdm(range(_N_GAMES)):
        for _ in range(_N_EPISODES):
            training()








def training():

    """
    n-step semi-gradient Sarsa algorithm
    for finding optimal q and pi via Linear
    FA with n-step TD updates.
    """
    # Create epsilon-greedy policy
    _POLICY = make_epsilon_greedy_policy()

    # Reset the environment and pick the first action
    state = _ENVIROMENT_CLASS.reset_env_approximate(_ENV)
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
            target = 0
            for i in range(update_time + 1, min(T, update_time + _N_STEP) + 1):
                target += np.power(_GAMMA, i - update_time - 1) * rewards[i]
            if update_time + _N_STEP < T:
                q_values_next = _ESTIMATOR.predict(states[update_time + _N_STEP])
                target += q_values_next[actions[update_time + _N_STEP]]

            # Update step
            _ESTIMATOR.update(states[update_time], actions[update_time], target)

        if update_time == T - 1:
            break

        state = next_state
        action = next_action

    ret = np.sum(rewards)

    return t, ret




def make_epsilon_greedy_policy():
    """
    Creates an epsilon-greedy policy based on a
    given q-value approximator and epsilon.
    """
    def policy_fn(observation):
        action_probs = np.ones(_ENVIROMENT_CLASS.number_actions(_ENV), dtype=float) * _EPSILON / _ENVIROMENT_CLASS.number_actions(_ENV)
        q_values = _ESTIMATOR.predict(observation)
        best_action_idx = np.argmax(q_values)
        action_probs[best_action_idx] += (1.0 - _EPSILON)
        return action_probs
    return policy_fn
