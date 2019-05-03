import numpy as np
from agents.tiles import *



def run_game(env, action):

    next_state, reward, done, info = env.step(action)

    state_adj = (next_state - env.observation_space.low)*np.array([10, 100])
    state_adj = np.round(state_adj, 0).astype(int)
    state_adj = state_adj[0] + 19*state_adj[1]

    return [state_adj, reward, done, info]

def run_game_approximate(env, action):
    return env.step(action)


def test_policy(env, action):

    next_state, reward, done, info = env.step(action)
    state_adj = (next_state - env.observation_space.low)*np.array([10, 100])
    state_adj = np.round(state_adj, 0).astype(int)
    state_adj = state_adj[0] + 19*state_adj[1]

    env_info = {
        "next_state": state_adj,
        "reward": reward,
        "done": done,
        "info": info
        }

    if done and reward == 1:
        return {"env_info": env_info, "average": reward}


    return {"env_info": env_info, "average": reward}



def type_test():

    return ["average"]

def number_states(env):
    '''
    Discretize the state space. One simple way in which this can be done
    is to round the first element of the state vector to the nearest 0.1
    and the second element to the nearest 0.01, and then (for convenience)
    multiply the first element by 10 and the second by 100.
    '''
    return list(range(0, 285))


def number_actions(env):
    return env.action_space.n

def reset_env(env):

    state = env.reset()

    state_adj = (state - env.observation_space.low)*np.array([10, 100])
    state_adj = np.round(state_adj, 0).astype(int)

    return state_adj[0] + 19*state_adj[1]


def reset_env_approximate(env):
    return env.reset()



class QEstimator():
    """
    Linear action-value (q-value) function approximator for
    semi-gradient methods with state-action featurization via tile coding.
    """

    def __init__(self, step_size, env, num_tilings=8, max_size=4096, tiling_dim=None, trace=False):

        self.env = env
        self.trace = trace
        self.max_size = max_size
        self.num_tilings = num_tilings
        self.tiling_dim = tiling_dim or num_tilings

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

        # Tilecoding software partitions at integer boundaries, so must rescale
        # position and velocity space to span tiling_dim x tiling_dim region.
        self.position_scale = self.tiling_dim / (env.observation_space.high[0] \
                                                  - env.observation_space.low[0])
        self.velocity_scale = self.tiling_dim / (env.observation_space.high[1] \
                                                  - env.observation_space.low[1])

    def featurize_state_action(self, state, action):
        """
        Returns the featurized representation for a
        state-action pair.
        """
        featurized = tiles(self.iht, self.num_tilings,
                           [self.position_scale * state[0],
                            self.velocity_scale * state[1]],
                           [action])
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
