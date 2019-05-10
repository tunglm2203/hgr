import threading

import numpy as np
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size_in_transitions = size_in_transitions  # Measure in number of transitions
        self.size = size_in_transitions // T    # Measure in number of episodes
        self.T = T
        self.sample_transitions = sample_transitions

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        self.lock = threading.Lock()
        self._next_idx = 0

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size):
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
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]

        self._next_idx = idx
        return idx


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_shapes, size_in_transitions, T, alpha,
                 replay_strategy, replay_k, reward_fun):
        super(PrioritizedReplayBuffer, self).__init__(buffer_shapes, size_in_transitions, T - 1, sample_transitions=None)

        if replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:  # 'replay_strategy' == 'none'
            self.future_p = 0
        self.reward_fun = reward_fun

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < self.size_in_transitions:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        super().store_episode(episode_batch)

        idx_ep = self._next_idx
        if isinstance(idx_ep, np.int64):
            batch_size = 1
            idx_ep = np.array([idx_ep])
        elif isinstance(idx_ep, np.ndarray):
            batch_size = idx_ep.shape[0]
        else:
            batch_size = None

        for k in range(batch_size):
            for i in range(self.T):
                idx = idx_ep[k] * self.T + i
                self._it_sum[idx] = self._max_priority ** self._alpha
                self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.get_current_size())
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = np.random.uniform() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def _encode_sample(self, episode_batch, idxes):
        T = episode_batch['u'].shape[1]
        # rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = len(idxes)

        # Compute episodes and time steps to use from indexes (idxes)
        episode_idxs = np.zeros(batch_size, dtype=np.int64)
        t_samples = np.zeros(batch_size, dtype=np.int64)
        for i in range(batch_size):
            episode_idxs[i] = int(idxes[i] // self.T)
            t_samples[i] = int(idxes[i] - (idxes[i] // self.T) * self.T)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}
        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = self.reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert (transitions['u'].shape[0] == batch_size)

        return transitions, [episode_idxs, t_samples]

    def sample(self, batch_size, beta):
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

        # PER: Sample from binary heap (containing indexes)
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.get_current_size()) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.get_current_size()) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights).squeeze()
        transitions, [episode_idxs, t_samples] = self._encode_sample(buffers, idxes)

        # Loop for checking
        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions, [episode_idxs, t_samples, weights, idxes]

    def sample_uniformly(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        return transitions, [episode_idxs, t_samples]

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

                sets priority of transition at index idxes[i] in buffer
                to priorities[i].

                Parameters
                ----------
                idxes: [int]
                    List of idxes of sampled transitions
                priorities: [float]
                    List of updated priorities corresponding to
                    transitions at the sampled idxes denoted by
                    variable `idxes`.
                """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.get_current_size()
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


def from_continous_idx_to_episode_idx(idxes, T):
    batch_size = len(idxes)

    # Compute episodes and time steps to use from indexes (idxes)
    episode_idxs = np.zeros(batch_size, dtype=np.int64)
    t_samples = np.zeros(batch_size, dtype=np.int64)
    for i in range(batch_size):
        episode_idxs[i] = int(idxes[i] // T)
        t_samples[i] = int(idxes[i] - (idxes[i] // T) * T)

    return episode_idxs, t_samples


def from_episode_idx_to_continous_idx(episode_idxs, t_samples, T):
    return episode_idxs * T + t_samples


