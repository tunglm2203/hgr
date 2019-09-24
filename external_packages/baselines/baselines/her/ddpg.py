from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from baselines import logger
from baselines.her.util import (
    import_function, flatten_grads, transitions_in_episode_batch, convert_episode_to_batch_major)
from baselines.her.normalizer import Normalizer
from baselines.her.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from baselines.common.mpi_adam import MpiAdam
from baselines.common import tf_util
from baselines.common.schedules import LinearSchedule


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class DDPG(object):
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class,
                 polyak, batch_size, q_lr, pi_lr, norm_eps, norm_clip, max_u,
                 action_l2, clip_obs, scope, time_horizon, rollout_batch_size,
                 subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 bc_loss, q_filter, num_demo, demo_batch_size, prm_loss_weight,
                 aux_loss_weight, sample_transitions, gamma, use_per, total_timesteps,
                 prioritized_replay_alpha, prioritized_replay_beta0,
                 prioritized_replay_beta_iters, prioritized_replay_alpha_prime,
                 prioritized_replay_beta0_prime, prioritized_replay_beta_iters_prime,
                 prioritized_replay_eps, use_huber_loss,
                 train_pi_interval, train_q_interval, info,
                 reuse=False):
        """
        Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).
        Added functionality to use demonstrations for training to overcome exploration problem.

        :param input_dims: ({str: int}) dimensions for the observation (o), the goal (g), and the actions (u)
        :param buffer_size: (int) number of transitions that are stored in the replay buffer
        :param hidden: (int) number of units in the hidden layers
        :param layers: (int) number of hidden layers
        :param network_class: (str) the network class that should be used (e.g. 'stable_baselines.her.ActorCritic')
        :param polyak: (float) coefficient for Polyak-averaging of the target network
        :param batch_size: (int) batch size for training
        :param q_lr: (float) learning rate for the Q (critic) network
        :param pi_lr: (float) learning rate for the pi (actor) network
        :param norm_eps: (float) a small value used in the normalizer to avoid numerical instabilities
        :param norm_clip: (float) normalized inputs are clipped to be in [-norm_clip, norm_clip]
        :param max_u: (float) maximum action magnitude, i.e. actions are in [-max_u, max_u]
        :param action_l2: (float) coefficient for L2 penalty on the actions
        :param clip_obs: (float) clip observations before normalization to be in [-clip_obs, clip_obs]
        :param scope: (str) the scope used for the TensorFlow graph
        :param time_horizon: (int) the time horizon for rollouts
        :param rollout_batch_size: (int) number of parallel rollouts per DDPG agent
        :param subtract_goals: (function (np.ndarray, np.ndarray): np.ndarray) function that subtracts goals from
        each other
        :param relative_goals: (boolean) whether or not relative goals should be fed into the network
        :param clip_pos_returns: (boolean) whether or not positive returns should be clipped
        :param clip_return: (float) clip returns to be in [-clip_return, clip_return]
        :param bc_loss: whether or not the behavior cloning loss should be used as an auxilliary loss
        :param q_filter: whether or not a filter on the q value update should be used when training with demonstartions
        :param num_demo: number of episodes in to be used in the demonstration buffer
        :param demo_batch_size: number of samples to be used from the demonstrations buffer, per mpi thread
        :param prm_loss_weight: weight corresponding to the primary loss
        :param aux_loss_weight: weight corresponding to the auxilliary loss also called the cloning loss
        :param sample_transitions: (function (dict, int): dict) function that samples from the replay buffer
        :param gamma: (float) gamma used for Q learning updates
        :param use_per: (boolean) whether or not use Prioritized Replay Buffer (PER)
        :param total_timesteps: (int) number of timesteps for training
        :param reuse: (boolean) whether or not the networks should be reused
        :param prioritized_replay_alpha: (float) alpha parameter for prioritized replay buffer
        :param prioritized_replay_beta0: (float) initial value of beta for prioritized replay buffer
        :param prioritized_replay_beta_iters: (int) number of iterations over which beta will be annealed from initial
        value to 1.0. If set to None equals to total_timesteps.
        :param prioritized_replay_eps: (float) epsilon to add to the TD errors when updating priorities
        :param info: (str) Name of the gym environment
        :param use_huber_loss: (boolean)
        """

        self.input_dims = input_dims
        self.buffer_size = ((buffer_size // time_horizon) // rollout_batch_size) * rollout_batch_size * time_horizon
        self.hidden = hidden
        self.layers = layers
        self.network_class = network_class
        self.polyak = polyak
        self.batch_size = batch_size
        self.q_lr = q_lr
        self.pi_lr = pi_lr
        self.norm_eps = norm_eps
        self.norm_clip = norm_clip
        self.max_u = max_u
        self.action_l2 = action_l2
        self.clip_obs = clip_obs
        self.scope = scope
        self.relative_goals = relative_goals
        self.time_horizon = time_horizon
        self.rollout_batch_size = rollout_batch_size
        self.subtract_goals = subtract_goals
        self.clip_pos_returns = clip_pos_returns
        self.clip_return = clip_return
        self.bc_loss = bc_loss
        self.q_filter = q_filter
        self.num_demo = num_demo
        self.demo_batch_size = demo_batch_size
        self.prm_loss_weight = prm_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.sample_transitions = sample_transitions
        self.gamma = gamma
        self.use_per = use_per
        self.total_timesteps = total_timesteps
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.prioritized_replay_alpha_prime = prioritized_replay_alpha_prime
        self.prioritized_replay_beta0_prime = prioritized_replay_beta0_prime
        self.prioritized_replay_beta_iters_prime = prioritized_replay_beta_iters_prime
        self.prioritized_replay_eps = prioritized_replay_eps
        self.use_huber_loss = use_huber_loss
        self.train_pi_interval = train_pi_interval
        self.train_q_interval = train_q_interval
        self.info = info
        self.reuse = reuse

        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class)

        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        stage_shapes['important_weight'] = (None,)
        self.stage_shapes = stage_shapes

        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(reuse=reuse)

        # Configure the replay buffer.
        buffer_shapes = {key: (self.time_horizon if key != 'o' else self.time_horizon + 1, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.time_horizon + 1, self.dimg)

        if self.prioritized_replay_beta_iters is None:
            self.prioritized_replay_beta_iters = self.total_timesteps

        self.beta_schedule = LinearSchedule(schedule_timesteps=self.prioritized_replay_beta_iters,
                                            initial_p=self.prioritized_replay_beta0,
                                            final_p=1.0)

        if self.use_per:
            self.buffer = PrioritizedReplayBuffer(buffer_shapes, self.buffer_size, self.time_horizon,
                                                  alpha=self.prioritized_replay_alpha,
                                                  alpha_prime=self.prioritized_replay_alpha_prime,
                                                  replay_strategy=self.sample_transitions['replay_strategy'],
                                                  replay_k=self.sample_transitions['replay_k'],
                                                  reward_fun=self.sample_transitions['reward_fun'])
            if self.bc_loss:
                # Initialize the demo buffer in the same way as the primary data buffer
                self.demo_buffer = PrioritizedReplayBuffer(buffer_shapes, self.buffer_size, self.time_horizon,
                                                           alpha=self.prioritized_replay_alpha,
                                                           alpha_prime=self.prioritized_replay_alpha_prime,
                                                           replay_strategy=self.sample_transitions['replay_strategy'],
                                                           replay_k=self.sample_transitions['replay_k'],
                                                           reward_fun=self.sample_transitions['reward_fun'])
        else:
            self.buffer = ReplayBuffer(buffer_shapes, self.buffer_size, self.time_horizon, self.sample_transitions)
            if self.bc_loss:
                # Initialize the demo buffer in the same way as the primary data buffer
                self.demo_buffer = ReplayBuffer(buffer_shapes, self.buffer_size,
                                                self.time_horizon, self.sample_transitions)

        self.episode_idxs = np.zeros(self.batch_size, dtype=np.int64)
        self.episode_idxs_for_bc = np.zeros(self.batch_size, dtype=np.int64)
        self.transition_idxs = np.zeros(self.batch_size, dtype=np.int64)
        self.transition_idxs_for_bc = np.zeros(self.batch_size, dtype=np.int64)

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def step(self, obs):
        actions = self.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goal'])
        return actions, None, None, None

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_q=False):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf]
        if compute_q:
            vals += [policy.Q_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (
                self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def init_demo_buffer(self, demo_data_file, update_stats=True):  # function that initializes the demo buffer

        demo_data = np.load(demo_data_file)  # load the demonstration data from data file
        info_keys = [key.replace('info_', '') for key in self.input_dims.keys() if key.startswith('info_')]
        info_values = [np.empty((self.time_horizon, 1, self.input_dims['info_' + key]), np.float32) for key in info_keys]

        demo_data_obs = demo_data['obs']
        demo_data_acs = demo_data['acs']
        demo_data_info = demo_data['info']

        # Initializing the whole demo buffer at the start of the training
        for epsd in range(self.num_demo):
            obs, acts, goals, achieved_goals = [], [], [], []
            i = 0
            # This loop is necessary since demontration data might have different format with episode
            for transition in range(self.time_horizon):
                obs.append([demo_data_obs[epsd][transition]['observation']])
                acts.append([demo_data_acs[epsd][transition]])
                goals.append([demo_data_obs[epsd][transition]['desired_goal']])
                achieved_goals.append([demo_data_obs[epsd][transition]['achieved_goal']])
                for idx, key in enumerate(info_keys):
                    info_values[idx][transition, i] = demo_data_info[epsd][transition][key]
            obs.append([demo_data_obs[epsd][self.time_horizon]['observation']])
            achieved_goals.append([demo_data_obs[epsd][self.time_horizon]['achieved_goal']])

            episode = dict(o=obs,
                           u=acts,
                           g=goals,
                           ag=achieved_goals)
            for key, value in zip(info_keys, info_values):
                episode['info_{}'.format(key)] = value

            episode = convert_episode_to_batch_major(episode)
            # create the observation dict and append them into the demonstration buffer
            self.demo_buffer.store_episode(episode)
            logger.debug("Demo buffer size currently ", self.demo_buffer.get_current_size())

            # add transitions to normalizer to normalize the demo data as well
            if update_stats:
                episode['o_2'] = episode['o'][:, 1:, :]
                episode['ag_2'] = episode['ag'][:, 1:, :]
                num_normalizing_transitions = transitions_in_episode_batch(episode)
                if self.use_per:
                    transitions, _ = PrioritizedReplayBuffer.sample_uniformly(episode, num_normalizing_transitions)
                else:
                    transitions, _ = self.sample_transitions(episode, num_normalizing_transitions)

                o, g, ag = transitions['o'], transitions['g'], transitions['ag']
                transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
                # No need to preprocess the o_2 and g_2 since this is only used for stats

                self.o_stats.update(transitions['o'])
                self.g_stats.update(transitions['g'])

                self.o_stats.recompute_stats()
                self.g_stats.recompute_stats()
            episode.clear()

        logger.info("Demo buffer size: ",
                    self.demo_buffer.get_current_size())  # print out the demonstration buffer size

    def store_episode(self, episode, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """
        self.buffer.store_episode(episode)

        # add transitions to normalizer
        if update_stats:
            # add transitions to normalizer
            episode['o_2'] = episode['o'][:, 1:, :]
            episode['ag_2'] = episode['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode)
            if self.use_per:
                transitions, _ = PrioritizedReplayBuffer.sample_uniformly(episode, num_normalizing_transitions)
            else:
                transitions, _ = self.sample_transitions(episode, num_normalizing_transitions)

            o, g, ag = transitions['o'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()

    def _grads_pi(self):
        # Avoid feed_dict here for performance!
        if self.bc_loss == 1:
            actor_loss, pi_grad, cloning_loss = self.sess.run([
                self.pi_loss_tf,
                self.pi_grad_tf,
                self.cloning_loss_tf
            ])
            return actor_loss, cloning_loss, pi_grad
        else:
            actor_loss, pi_grad = self.sess.run([
                self.pi_loss_tf,
                self.pi_grad_tf,
            ])
            return actor_loss, pi_grad

    def _grads_q(self):
        # Avoid feed_dict here for performance!
        if self.bc_loss == 1:
            td_error, critic_loss, q_grad = self.sess.run([
                self.td_error_tf,
                self.Q_loss_tf,
                self.Q_grad_tf,
            ])
            return td_error, critic_loss, q_grad
        else:
            td_error, critic_loss, q_grad = self.sess.run([
                self.td_error_tf,
                self.Q_loss_tf,
                self.Q_grad_tf,
            ])
            return td_error, critic_loss, q_grad

    def _update_pi(self, pi_grad):
        self.pi_adam.update(pi_grad, self.pi_lr)

    def _update_q(self, q_grad):
        self.Q_adam.update(q_grad, self.q_lr)

    def sample_batch(self, time_step=None):
        if self.use_per:
            if self.bc_loss:  # use demonstration buffer to sample
                # extra_infor: [episode_idxs, transition_idxs, weights]
                transitions, extra_info = self.buffer.sample(self.batch_size - self.demo_batch_size,
                                                             beta=self.beta_schedule.value(time_step),
                                                             beta_prime=self.beta_schedule.value(time_step))
                episode_idxs, transition_idxs, weights = extra_info

                transitions_demo, extra_info_demo = self.demo_buffer.sample(self.demo_batch_size,
                                                                            beta=self.beta_schedule.value(time_step),
                                                                            beta_prime=self.beta_schedule.value(time_step))
                episode_idxs_for_bc, transition_idxs_for_bc, weights_for_bc = extra_info_demo

                for k, values in transitions_demo.items():
                    rollout_v = transitions[k].tolist()
                    for v in values:
                        rollout_v.append(v.tolist())
                    transitions[k] = np.array(rollout_v)

                extra_info = extra_info + extra_info_demo
                transitions['important_weight'] = np.concatenate((weights, weights_for_bc), axis=0)
            else:
                transitions, extra_info = self.buffer.sample(self.batch_size, beta=self.beta_schedule.value(time_step),
                                                             beta_prime=self.beta_schedule.value(time_step))
                weights = extra_info[2]
                transitions['important_weight'] = weights
        else:
            if self.bc_loss:  # use demonstration buffer to sample
                transitions, extra_info = self.buffer.sample(self.batch_size - self.demo_batch_size)
                episode_idxs, t_samples = extra_info

                transitions_demo, extra_info = self.demo_buffer.sample(self.demo_batch_size)
                episode_idxs_1, t_samples_1 = extra_info

                for k, values in transitions_demo.items():
                    rollout_v = transitions[k].tolist()
                    for v in values:
                        rollout_v.append(v.tolist())
                    transitions[k] = np.array(rollout_v)

                episode_idxs = np.concatenate((episode_idxs, episode_idxs_1), axis=0)
                t_samples = np.concatenate((t_samples, t_samples_1), axis=0)
                extra_info = [episode_idxs, t_samples]
            else:
                transitions, extra_info = self.buffer.sample(self.batch_size)
            transitions['important_weight'] = np.ones(self.batch_size)

        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]

        return transitions_batch, extra_info

    def stage_batch(self, batch=None, time_step=None):
        if batch is None:
            batch, extra_info = self.sample_batch(time_step=time_step)
            if self.use_per:
                # Don't get important weight `weights` here since it is already included in `batch`
                self.episode_idxs = extra_info[0]
                self.transition_idxs = extra_info[1]
                if self.bc_loss:
                    self.episode_idxs_for_bc = extra_info[3]
                    self.transition_idxs_for_bc = extra_info[4]

        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def train(self, stage=True, time_step=None, train_q=True, train_pi=True):
        # Sampling batch of data
        batch, extra_info = self.sample_batch(time_step=time_step)
        if self.use_per:
            self.episode_idxs = extra_info[0]
            self.transition_idxs = extra_info[1]
            if self.bc_loss:
                self.episode_idxs_for_bc = extra_info[3]
                self.transition_idxs_for_bc = extra_info[4]

        td_error, critic_loss, actor_loss, cloning_loss = 0., 0., 0., 0.
        q_grad, pi_grad = None, None
        if self.bc_loss == 1:
            if train_q:
                if stage:
                    self.stage_batch(batch=batch)
                td_error, critic_loss, q_grad = self._grads_q()
            if train_pi:
                if stage:
                    self.stage_batch(batch=batch)
                actor_loss, cloning_loss, pi_grad = self._grads_pi()
            if train_q:
                self._update_q(q_grad)
            if train_pi:
                self._update_pi(pi_grad)
            return td_error, critic_loss, actor_loss, cloning_loss
        else:
            if train_q:
                if stage:
                    self.stage_batch(batch=batch)
                td_error, critic_loss, q_grad = self._grads_q()
            if train_pi:
                if stage:
                    self.stage_batch(batch=batch)
                actor_loss, pi_grad = self._grads_pi()
            if train_q:
                self._update_q(q_grad)
            if train_pi:
                self._update_pi(pi_grad)
            return td_error, critic_loss, actor_loss, None

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def _create_network(self, reuse=False):
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))
        self.sess = tf_util.get_session()

        # running averages
        with tf.variable_scope('o_statistic') as scope:
            if reuse:
                scope.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_statistic') as scope:
            if reuse:
                scope.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i]) for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        # Mask used to choose the demo buffer samples
        if self.bc_loss == 1:
            mask = np.concatenate((np.zeros(self.batch_size - self.demo_batch_size), np.ones(self.demo_batch_size)),
                                  axis=0)

        # networks
        with tf.variable_scope('main') as scope:
            if reuse:
                scope.reuse_variables()
            self.main = self.create_actor_critic(inputs_tf=batch_tf, dimo=self.dimo, dimg=self.dimg,
                                                 dimu=self.dimu, max_u=self.max_u,
                                                 o_stats=self.o_stats, g_stats=self.g_stats,
                                                 hidden=self.hidden, layers=self.layers)
            scope.reuse_variables()
        with tf.variable_scope('target') as scope:
            if reuse:
                scope.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = self.create_actor_critic(target_batch_tf, dimo=self.dimo, dimg=self.dimg,
                                                   dimu=self.dimu, max_u=self.max_u,
                                                   o_stats=self.o_stats, g_stats=self.g_stats,
                                                   hidden=self.hidden, layers=self.layers)
            scope.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        # loss functions
        target_q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)

        with tf.variable_scope('loss') as scope:
            if reuse:
                scope.reuse_variables()

            # Loss for critic
            target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_q_pi_tf, *clip_range, name='target_tf')
            self.td_error_tf = tf.stop_gradient(target_tf) - self.main.Q_tf
            if self.use_huber_loss:
                errors = tf_util.huber_loss(self.td_error_tf)
            else:
                errors = tf.square(self.td_error_tf)

            self.Q_loss_tf = tf.reduce_mean(errors * batch_tf['important_weight'])

            # Loss for actor
            if self.bc_loss == 1 and self.q_filter == 1:
                # Choosing samples where the demonstrator's action better than actor's action according to the critic
                mask_main = tf.reshape(tf.boolean_mask(self.main.Q_tf > self.main.Q_pi_tf, mask), [-1])
                # define the cloning loss on the actor's actions only on the samples which adhere to the above masks
                self.cloning_loss_tf = tf.reduce_sum(
                    tf.square(
                        tf.boolean_mask(tf.boolean_mask(self.main.pi_tf, mask), mask_main, axis=0) -
                        tf.boolean_mask(tf.boolean_mask(batch_tf['u'], mask), mask_main, axis=0)
                    ), name='cloning_loss_tf'
                )

                self.pi_loss_tf = -self.prm_loss_weight * tf.reduce_mean(self.main.Q_pi_tf)
                self.pi_loss_tf += self.prm_loss_weight * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
                self.pi_loss_tf += self.aux_loss_weight * self.cloning_loss_tf

            elif self.bc_loss == 1 and self.q_filter == 0:  # train with demonstrations without q_filter
                self.cloning_loss_tf = tf.reduce_sum(
                    tf.square(tf.boolean_mask(self.main.pi_tf, mask) - tf.boolean_mask((batch_tf['u']), mask)),
                    name='cloning_loss_tf')
                self.pi_loss_tf = -self.prm_loss_weight * tf.reduce_mean(self.main.Q_pi_tf)
                self.pi_loss_tf += self.prm_loss_weight * self.action_l2 * tf.reduce_mean(
                    tf.square(self.main.pi_tf / self.max_u))
                self.pi_loss_tf += self.aux_loss_weight * self.cloning_loss_tf

            else:  # If  not training with demonstrations
                self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
                self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
            scope.reuse_variables()

        q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'), name='q_gradient')
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'), name='pi_gradient')
        assert len(self._vars('main/Q')) == len(q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.Q_grad_tf = flatten_grads(grads=q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # polyak averaging
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]),
                zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """
        Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        # This list below to exclude variables from saving
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        _state = state.copy()
        if state['use_per']:
            _state['sample_transitions'] = {
                'replay_strategy': None,
                'replay_k': None,
                'reward_fun': None
            }

        excluded_subnames = ['dimo', 'dimg', 'dimu', 'episode_idxs', 'episode_idxs_for_bc',
                             'transition_idxs', 'transition_idxs_for_bc', 'tf', 'beta_schedule']
        for key in excluded_subnames:
            del _state[key]

        self.__init__(**_state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars_list = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars_list) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars_list, state["tf"])]
        self.sess.run(node)

    def save(self, save_path):
        tf_util.save_variables(save_path, sess=self.sess)
