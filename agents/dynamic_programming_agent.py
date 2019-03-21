import numpy as np
import random
from tqdm import tqdm
import sys
sys.path.insert(0, 'enviroments')

import enviroment_choose


def run_agent(env, n_games, n_episodes, epsilon=0.01):

    global enviroment_class
    enviroment_class = enviroment_choose.env_choose(env)
    #return policy_iteration(env, n_games, n_episodes, epsilon)
    return policy_iteration(env, n_games)


def policy_iteration(env, n_games, discount_factor=1.0, max_iterations=1e9):

    tests_result = []

    # Start with a random policy
    policy = enviroment_class.create_random_policy(env)
    agent_info = {
        "policy": policy,
        "state_action_table": enviroment_class.create_state_action_dictionary(env, policy),
        "returns_number": {}
    }

    # Repeat until convergence or critical number of iterations reached
    for i in tqdm(range(int(1000))):

        # Evaluate current policy
        V = policy_evaluation(env, policy, discount_factor=discount_factor)
        new_policy = policy_improvement(env, V)

        # Stop if the policy is unchanged after an improvement step
        if (new_policy == policy):
            break;

        # Stop if the value function estimates for successive policies has converged
        #if np.max(abs(policy_evaluation(env, policy) - policy_evaluation(env, new_policy))) < theta*1e2:
        #    break;

        policy = new_policy


    return policy, V






def policy_improvement(env, V, gamma=1):

    # Start with a random policy
    policy = enviroment_class.create_random_policy(env)

    for state in policy.keys():

        for action in policy[state].keys():
            policy[state][action] = 0

        best_action = np.argmax(policy[state])
        policy[state][best_action] = 1

    return policy




def policy_evaluation(env, policy, discount_factor=1.0, theta=1e-8, max_iterations=1e9):
        # Initialize a value function for each state as zero
        V = [0 for i in policy.keys()]
        # Repeat until change in value is below the threshold
        for i in range(int(max_iterations)):
                # Initialize a change of value function as zero
                delta = 0
                # Iterate though each state
                for state in policy.keys():
                       # Initial a new value of current state
                       v = 0
                       # Try all possible actions which can be taken from this state
                       for action in policy[state]:
                           action_probability = policy[state][action]
                           # Check how good next state will be
                           for state_probability, next_state, reward, terminated in env.env.P[state][action]:
                               # Calculate the expected value
                               v += action_probability * state_probability * (reward + discount_factor * V[next_state])

                       delta = max(delta, np.abs(V[state] - v)) # Calculate the absolute change of value function
                       V[state] = v # Update value function

                # Terminate if value change is insignificant
                if delta < theta:
                        return V


def one_step_lookahead(env, policy, state, V, discount_factor):
        action_values = np.zeros(len(policy[state]))
        for action in range(len(policy[state])):
                for probability, next_state, reward, terminated in env.env.P[state][action]:
                        action_values[action] += probability * (reward + discount_factor * V[next_state])
        return action_values














'''
def value_iteration(environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
        # Initialize state-value function with zeros for each environment state
        V = np.zeros(environment.nS)
        for i in range(int(max_iterationsations)):
                # Early stopping condition
                delta = 0
                # Update each state
                for state in range(environment.nS):
                        # Do a one-step lookahead to calculate state-action values
                        action_value = one_step_lookahead(environment, state, V, discount_factor)
                        # Select best action to perform based on the highest state-action value
                        best_action_value = np.max(action_value)
                        # Calculate change in value
                        delta = max(delta, np.abs(V[state] - best_action_value))
                        # Update the value function for current state
                        V[state] = best_action_value
                        # Check if we can stop
                if delta < theta:
                        print(f'Value-iteration converged at iteration#{i}.')
                        break

        # Create a deterministic policy using the optimal value function
        policy = np.zeros([environment.nS, environment.nA])
        for state in range(environment.nS):
                # One step lookahead to find the best action for this state
                action_value = one_step_lookahead(environment, state, V, discount_factor)
                # Select best action based on the highest state-action value
                best_action = np.argmax(action_value)
                # Update the policy to perform a better action at a current state
                policy[state, best_action] = 1.0
        return policy, V
'''
