import random
import numpy as np
from her import make_sample_her_transitions
import gym
import threading
from collections import OrderedDict

CACHED_ENVS = {}


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class HindsightExperientReplay(object):
    def __init__(self, size, env_name='FetchPickAndPlace-v1', replay_strategy='future', replay_k=4):
        """
        Create HindsightExperientReplay buffer.
        NOTE: The transitions in episode including reset, so if horizon is T, the number of
        observations is T+1, number of (actions, rewards, infos is T).

        :param size: (int)  Max number of transitions to store in the buffer.
        When the buffer overflows the old memories are dropped.
        """

        def make_env():
            return gym.make(env_name)
        env = cached_make_env(make_env)

        self.horizon = env._max_episode_steps
        self._maxsize = size  # Measured by number of transitions
        self._next_idx_in_episode = 0
        self._next_episode_idx = 0
        self.current_n_episodes = 0
        self._max_episode = self._maxsize // self.horizon     # Measured by number of episodes

        self.sample_transitions = configure_her(make_env, replay_strategy, replay_k)
        # Refer from HER
        dims = configure_dims(make_env)

        # Obtain shape of observation
        input_shapes = dims_to_shapes(dims)
        self.dimo = dims['o']
        self.dimg = dims['g']
        self.dimu = dims['u']

        # Configure the replay buffer.
        buffer_shapes = {key: (self.horizon if key != 'o' else self.horizon + 1, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.horizon + 1, self.dimg)

        # dict_keys of self._storage: ['o', 'u', 'g', 'done', 'info_is_success', 'ag']
        self._storage = {key: np.empty([self._max_episode, *shape]) for key, shape in buffer_shapes.items()}

    def __len__(self):
        return self.current_n_episodes * self.horizon

    def add(self, obs_t, desired_goal, achieved_goal, action, done, info):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param desired_goal:
        :param achieved_goal:
        :param info:
        """

        if self._next_idx_in_episode == self.horizon:   # End of episode
            self._storage['o'][self._next_episode_idx][self._next_idx_in_episode] = obs_t
            self._storage['ag'][self._next_episode_idx][self._next_idx_in_episode] = achieved_goal
            # self._storage['done'][self._next_episode_idx][self._next_idx_in_episode] = done
            self._next_idx_in_episode = 0
            self._next_episode_idx = (self._next_episode_idx + 1) % self._max_episode
            if self.current_n_episodes < self._max_episode:
                self.current_n_episodes += 1
        else:
            self._storage['o'][self._next_episode_idx][self._next_idx_in_episode] = obs_t
            self._storage['u'][self._next_episode_idx][self._next_idx_in_episode] = action
            self._storage['g'][self._next_episode_idx][self._next_idx_in_episode] = desired_goal
            self._storage['ag'][self._next_episode_idx][self._next_idx_in_episode] = achieved_goal
            self._storage['done'][self._next_episode_idx][self._next_idx_in_episode] = done
            if 'is_success' in info:
                self._storage['info_is_success'][self._next_episode_idx][self._next_idx_in_episode] = info['is_success']
            self._next_idx_in_episode += 1

    def sample(self, batch_size):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        # Get all data from self._storage buffer
        buffers = {}
        for key in self._storage.keys():
            buffers[key] = self._storage[key][:self.current_n_episodes]

        # Create next observation and next achieved goal
        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        # keys of a transition: dict_keys(['o', 'u', 'g', 'info_is_success', 'ag', 'o_2', 'ag_2', 'r'])
        transitions = self.sample_transitions(buffers, batch_size)
        for key in (['r', 'o_2', 'ag_2'] + list(self._storage.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        transitions['g_2'] = transitions['g']
        obses_t = np.concatenate((np.array(transitions['o']), np.array(transitions['g'])), axis=1)
        actions = np.array(transitions['u'])
        rewards = np.array(transitions['r'])
        obses_tp1 = np.concatenate((np.array(transitions['o_2']), np.array(transitions['g_2'])), axis=1)
        dones = np.array(transitions['done'])

        return obses_t, actions, rewards, obses_tp1, dones

    def get_last_episodes(self, n, key):
        """
        Get n last episodes in buffer
        :param n: Number of episodes want to get
        :param key: Should be one of ['o', 'u', 'g', 'done', 'info_is_success', 'ag']
        :return:
        """
        assert key == 'o' or key == 'u' or key == 'g' or key == 'done' or key == 'info_is_success' or key == 'ag', \
            print('[AIM-ERROR] Do not support this key: %s' % key)
        if self.current_n_episodes > n:
            return self._storage[key][self._next_episode_idx-n:self._next_episode_idx]
        else:
            return None


def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]


def configure_her(make_env, replay_strategy='future', replay_k=4):

    env = cached_make_env(make_env)
    env.reset()

    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    # Prepare configuration for HER.
    her_params = {
        'reward_fun': reward_fun,
        'replay_strategy': replay_strategy,
        'replay_k': replay_k
    }
    sample_her_transitions = make_sample_her_transitions(replay_strategy=her_params['replay_strategy'],
                                                         replay_k=her_params['replay_k'],
                                                         reward_fun=her_params['reward_fun'])
    return sample_her_transitions


def configure_dims(make_env):
    env = cached_make_env(make_env)
    env.reset()
    obs, _, done, info = env.step(env.action_space.sample())

    dims = {
        'o': obs['observation'].shape[0],
        'u': env.action_space.shape[0],
        'g': obs['desired_goal'].shape[0],
        'done': done
    }

    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims['info_{}'.format(key)] = value.shape[0]
    return dims
