import numpy as np
from agents.tiles import *



def run_game_approximate(env, action):
    return env.step(action)



def test_policy_approximate(env, action):

    next_state, reward, done, info = env.step(action)

    env_info = {
        "next_state": next_state,
        "reward": reward,
        "done": done,
        "info": info
        }

    return {"env_info": env_info, "average": reward}



def type_test():

    return ["average"]


def number_actions(env):
    return env.action_space.n


def reset_env_approximate(env):
    return env.reset()

def probability(env):
    return None


class QEstimator():
    """
    Linear action-value (q-value) function approximator for
    semi-gradient methods with state-action featurization via tile coding.
    """

    def __init__(self, step_size, env, num_tilings=32, max_size=16777216, tiling_dim=None, trace=False):

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
        self.cos_theta1 = self.tiling_dim / (env.observation_space.high[0] \
                                                  - env.observation_space.low[0])
        self.sin_theta1 = self.tiling_dim / (env.observation_space.high[1] \
                                                  - env.observation_space.low[1])
        self.cos_theta2 = self.tiling_dim / (env.observation_space.high[2] \
                                                  - env.observation_space.low[2])
        self.sin_theta2 = self.tiling_dim / (env.observation_space.high[3] \
                                                  - env.observation_space.low[3])
        self.thetaDot1 = self.tiling_dim / (env.observation_space.high[4] \
                                                  - env.observation_space.low[4])
        self.thetaDot2 = self.tiling_dim / (env.observation_space.high[5] \
                                                  - env.observation_space.low[5])


    def featurize_state_action(self, state, action):
        """
        Returns the featurized representation for a
        state-action pair.
        """
        featurized = tiles(self.iht, self.num_tilings,
                           [self.cos_theta1 * state[0],
                            self.sin_theta1 * state[1],
                            self.cos_theta2 * state[2],
                            self.sin_theta2 * state[3],
                            self.thetaDot1 * state[4],
                            self.thetaDot2 * state[5]],
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
