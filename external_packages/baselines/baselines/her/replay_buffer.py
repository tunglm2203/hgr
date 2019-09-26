import threading

import numpy as np
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
import tensorflow as tf
from scipy.stats import multinomial


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, time_horizon, sample_transitions=None):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            time_horizon (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size_in_episodes = size_in_transitions // time_horizon    # Measure in number of episodes
        self.size_in_transitions = (size_in_transitions // time_horizon)*time_horizon  # Measure in num of transitions
        self.time_horizon = time_horizon
        self.sample_transitions = sample_transitions

        # self.buffers is {key: array(size_in_episodes x time_horizon or time_horizon+1 x dim_key)}
        self.buffers = {}
        for key, shape in buffer_shapes.items():
            self.buffers[key] = np.empty([self.size_in_episodes, *shape])

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        self.lock = threading.Lock()
        self._next_idx = 0

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size_in_episodes

    def sample(self, batch_size, beta=0., beta_prime=0.):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        transitions, [episode_idxs, t_samples] = self.sample_transitions(buffers, batch_size)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions, [episode_idxs, t_samples]

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (time_horizon or time_horizon+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.time_horizon
        self._next_idx = idxs

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.time_horizon

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size_in_episodes, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size_in_episodes:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size_in_episodes:
            overflow = inc - (self.size_in_episodes - self.current_size)
            idx_a = np.arange(self.current_size, self.size_in_episodes)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size_in_episodes, inc)

        # update replay size
        self.current_size = min(self.size_in_episodes, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_shapes, size_in_transitions, time_horizon, alpha, alpha_prime,
                 replay_strategy, replay_k, reward_fun):
        it_capacity = 1     # Iterator for computing capacity of buffer
        size_in_episodes = size_in_transitions // time_horizon
        while it_capacity < size_in_episodes:
            it_capacity *= 2
        size_in_transitions = it_capacity * time_horizon
        super(PrioritizedReplayBuffer, self).__init__(buffer_shapes, size_in_transitions, time_horizon)

        if replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:  # 'replay_strategy' == 'none'
            self.future_p = 0
        self.reward_fun = reward_fun

        assert alpha >= 0 and alpha_prime >= 0
        self._alpha = alpha
        self._alpha_prime = alpha_prime

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)

        # size of weight_of_episode is size_in_episodes x time_horizon^2 (size_in_episodes x 2500)
        # i-th 50 elements are i-th state combined with 50 goals from ag_1, ..., ag_49
        # [s0||ag_1, ..., s0||ag_50], [s1||ag_2, ..., s1||ag_50], ..., [s49||ag_50]
        #  \______50 elements_____/    \______49 elements_____/        \1 elements/
        #   \_______________________ 1275 elements ______________________________/
        # NOTE: s, ag is zero-based
        if self.replay_strategy == 'future':
            self._length_weight = int((self.time_horizon - 1) * self.time_horizon / 2)
            self.weight_of_transition = np.empty([self.size_in_episodes, self._length_weight])
            self._idx_state_and_future = np.empty(self._length_weight, dtype=list)
            # self._idx_state_and_future = np.empty([self._length_weight, 2], dtype=np.int32)  # Lookup table
            _idx = 0
            for i in range(self.time_horizon - 1):
                for j in range(i, self.time_horizon - 1):
                    self._idx_state_and_future[_idx] = [i, j + 1]
                    # self._idx_state_and_future[_idx, 0] = i
                    # self._idx_state_and_future[_idx, 1] = j + 1
                    _idx += 1
        elif self.replay_strategy == 'final':
            self._length_weight = int(self.time_horizon - 1)
            self.weight_of_transition = np.empty([self.size_in_episodes, self._length_weight])
            self._idx_state_and_future = np.empty([self._length_weight, 2], dtype=np.int32)  # Lookup table
            for i in range(self.time_horizon - 1):
                self._idx_state_and_future[i, 0] = i
                self._idx_state_and_future[i, 1] = self.time_horizon - 1

        self._max_episode_priority = 1.0
        self._max_transition_priority = 1.0

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (time_horizon or time_horizon+1) x dim_key)
        """
        super().store_episode(episode_batch)

        idx_ep = self._next_idx
        if isinstance(idx_ep, np.int64):
            rollout_batch_size = 1
            idx_ep = np.array([idx_ep])
        elif isinstance(idx_ep, np.ndarray):
            rollout_batch_size = idx_ep.shape[0]
        else:
            rollout_batch_size = None

        for k in range(rollout_batch_size):
            idx = idx_ep[k]
            self._it_sum[idx] = self._max_episode_priority ** self._alpha
            self._it_min[idx] = self._max_episode_priority ** self._alpha

        self.weight_of_transition[idx_ep] = \
            (np.ones((rollout_batch_size, self._length_weight)) * self._max_transition_priority) ** self._alpha_prime

    def sample(self, batch_size, beta=0., beta_prime=0.):
        """
        Returns:
             A dict {key: array(batch_size x shapes[key])}
             A list of:
                episode_idxs: Indexes of sampled episodes
                t_samples: Indexes of sampled transitions in episode format
                weights: Important weight corresponding to each transition
                idxes: Indexes of sampled transitions in continuous format
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        assert beta > 0
        assert beta_prime > 0

        # (1) Sampling in episode-level
        episode_idxs = self._sample_proportional(batch_size)
        if max(episode_idxs) >= self.get_current_episode_size():
            print('Error')
            import pdb; pdb.set_trace()
        assert max(episode_idxs) < self.get_current_episode_size(), 'Index out of range: {}'.format(max(episode_idxs))

        weight_of_episodes = []
        for idx in episode_idxs:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.get_current_episode_size()) ** (-beta)
            weight_of_episodes.append(weight)
        weight_of_episodes = np.array(weight_of_episodes).squeeze()

        # (2) Sampling in transition-level
        transitions, extra_info = self._encode_sample(buffers, episode_idxs, beta_prime)
        weight_of_transitions, transition_idxs = extra_info[:2]
        weights = weight_of_episodes * weight_of_transitions
        _max_weight = weights.max()
        weights = weights / _max_weight

        # Loop for asserting
        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions, [episode_idxs, transition_idxs, weights]

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.get_current_episode_size())
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = np.random.uniform() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return np.array(res)

    def _encode_sample(self, episode_batch, episode_idxs, beta_prime):
        batch_size = len(episode_idxs)

        transition_idxs = np.zeros(batch_size, dtype=np.int64)  # composed by `t_states` & `t_futures`
        t_states = np.zeros(batch_size, dtype=np.int64)
        t_futures = np.zeros(batch_size, dtype=np.int64)

        weight_of_transitions = np.zeros(batch_size, dtype=np.float)
        for i in range(batch_size):
            # _max_weight_transition = \
            #     (self.weight_of_transition[episode_idxs[i]].min() * self._length_weight) ** (-beta_prime)
            weight_prob = \
                self.weight_of_transition[episode_idxs[i]] / self.weight_of_transition[episode_idxs[i]].sum()

            # Compute timestep to use by sampling with probability from weight_prob
            _idx = multinomial.rvs(n=1, p=weight_prob).argmax()
            transition_idxs[i] = _idx
            t_states[i] = self._idx_state_and_future[_idx][0]   # Get index from lookup table
            t_futures[i] = self._idx_state_and_future[_idx][1]  # Get index from lookup table
            # weight_of_transitions[i] = \
            #     (self.weight_of_transition[episode_idxs[i], _idx] * self._length_weight) ** (-beta_prime) \
            #     / _max_weight_transition
            weight_of_transitions[i] = \
                (self.weight_of_transition[episode_idxs[i], _idx] * self._length_weight) ** (-beta_prime)

        transitions = {key: episode_batch[key][episode_idxs, t_states].copy()
                       for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], t_futures[her_indexes]]
        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = self.reward_fun(achieved_goal=reward_params['ag_2'],
                                           desired_goal=reward_params['g'],
                                           info=reward_params['info'])

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert (transitions['u'].shape[0] == batch_size)

        return transitions, [weight_of_transitions, transition_idxs]

    @staticmethod
    def sample_uniformly(episode_batch, batch_size_in_transitions):
        time_horizon = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(time_horizon, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        return transitions, [episode_idxs, t_samples]

    def update_priorities(self, episode_idxs, priorities, idxes_in_ep):
        """
        Update priorities of sampled transitions.
        sets priority of transition at episode episode_idxs[i], index idxes_in_ep[i] in buffer to priorities[i]
        :param episode_idxs: [int] List of indexes of sampled episodes
        :param priorities: List of updated priorities corresponding to transitions at the sampled indexes denoted by
        variable `episode_idxs` and `idxes_in_ep`.
        :param idxes_in_ep: [int] List of indexes of sampled transitions
        """
        assert len(episode_idxs) == len(priorities) and len(episode_idxs) == len(idxes_in_ep)
        for ep_idx, _priority_of_transition, transition_idx in zip(episode_idxs, priorities, idxes_in_ep):
            assert _priority_of_transition > 0
            assert 0 <= ep_idx < self.get_current_episode_size()
            # Update weight for transitions in 1 episode
            self.weight_of_transition[ep_idx, transition_idx] = _priority_of_transition ** self._alpha_prime

            # Update weight for all episodes
            _priority_of_episode = self.weight_of_transition[ep_idx].mean()
            self._it_sum[ep_idx] = _priority_of_episode ** self._alpha
            self._it_min[ep_idx] = _priority_of_episode ** self._alpha

            self._max_episode_priority = max(self._max_episode_priority, _priority_of_episode)
            self._max_transition_priority = max(self._max_transition_priority, _priority_of_transition)


def vmultinomial_sampling(counts, pvals, seed=None):
    k = tf.shape(pvals)[1]
    logits = tf.expand_dims(tf.log(pvals), 1)

    def sample_single(args):
        logits_, n_draw_ = args[0], args[1]
        xx = tf.multinomial(logits_, n_draw_, seed)
        indices = tf.cast(tf.reshape(xx, [-1, 1]), tf.int32)
        updates = tf.ones(n_draw_)  # tf.shape(indices)[0]
        return tf.scatter_nd(indices, updates, [k])

    x = tf.map_fn(sample_single, [logits, counts], dtype=tf.float32)

    return x
