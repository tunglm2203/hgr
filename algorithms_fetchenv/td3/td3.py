from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from baselines import logger
from util import (import_function, store_args, flatten_grads, transitions_in_episode_batch)
from normalizer import Normalizer
from replay_buffer import ReplayBuffer
from baselines.common.mpi_adam import MpiAdam
from baselines.her.util import convert_episode_to_batch_major
from tqdm import tqdm
import pickle


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


global demoBuffer


class TD3(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns,  clip_return, bc_loss, q_filter, num_demo,
                 sample_transitions, gamma, reuse=False, **kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """

        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class)

        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        # self.demo_batch_size = 128
        self.lambda1 = self.prm_loss_weight
        self.lambda2 = self.aux_loss_weight
        # self.l2_reg_coeff = 0.005

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        # for key in ['o', 'g']:
        #     stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['o_2'] = stage_shapes['o']
        stage_shapes['g_2'] = stage_shapes['g']
        stage_shapes['r'] = (None,)
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
        buffer_shapes = {key: (self.T if key != 'o' else self.T+1, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T+1, self.dimg)

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)
        global demoBuffer
        demoBuffer = ReplayBuffer(buffer_shapes, self.num_demo * self.rollout_batch_size * self.T, self.T, self.sample_transitions)

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

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.q1_pi_tf]
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
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def initDemoBuffer(self, demoDataFile, update_stats=True, load_from_pickle=False, pickle_file=''):

        if not load_from_pickle:
            demoData = np.load(demoDataFile)
            info_keys = [key.replace('info_', '') for key in self.input_dims.keys() if key.startswith('info_')]
            info_values = [np.empty((self.T, 1, self.input_dims['info_' + key]), np.float32) for key
                           in info_keys]    # tung: modify demo buffer

            for epsd in tqdm(range(self.num_demo)):
                obs, acts, goals, achieved_goals, dones = [], [], [], [], []
                i = 0
                done_ep = np.zeros((1, 1))
                for transition in range(self.T):
                    obs.append([demoData['obs'][epsd ][transition].get('observation')])
                    acts.append([demoData['acs'][epsd][transition]])
                    goals.append([demoData['obs'][epsd][transition].get('desired_goal')])
                    achieved_goals.append([demoData['obs'][epsd][transition].get('achieved_goal')])
                    done_ep[0] = [demoData['done'][epsd][transition]]
                    dones.append(done_ep.copy())
                    for idx, key in enumerate(info_keys):
                        info_values[idx][transition, i] = demoData['info'][epsd][transition][key]

                obs.append([demoData['obs'][epsd][self.T].get('observation')])
                achieved_goals.append([demoData['obs'][epsd][self.T].get('achieved_goal')])

                episode = dict(o=obs,
                               u=acts,
                               g=goals,
                               ag=achieved_goals,
                               d=dones)
                for key, value in zip(info_keys, info_values):
                    episode['info_{}'.format(key)] = value
                episode = convert_episode_to_batch_major(episode)
                global demoBuffer
                demoBuffer.store_episode(episode)

                print("Demo buffer size currently ", demoBuffer.get_current_size())

                if update_stats:
                    # add transitions to normalizer to normalize the demo data as well
                    episode['o_2'] = episode['o'][:, 1:, :]
                    episode['ag_2'] = episode['ag'][:, 1:, :]
                    num_normalizing_transitions = transitions_in_episode_batch(episode)
                    transitions = self.sample_transitions(episode, num_normalizing_transitions)

                    o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
                    transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
                    # No need to preprocess the o_2 and g_2 since this is only used for stats

                    self.o_stats.update(transitions['o'])
                    self.g_stats.update(transitions['g'])

                    self.o_stats.recompute_stats()
                    self.g_stats.recompute_stats()
                episode.clear()
            with open('demo_pickandplace.pkl', 'wb') as f:
                data = {
                    'current_size': demoBuffer.current_size,
                    'n_transitions_stored': demoBuffer.n_transitions_stored,
                    'buffers': demoBuffer.buffers,
                }
                pickle.dump(data, f)
        else:
            pickle_in = open(pickle_file, "rb")
            _data = pickle.load(pickle_in)
            # global demoBuffer
            demoBuffer.current_size = _data['current_size']
            demoBuffer.n_transitions_stored = _data['n_transitions_stored']
            demoBuffer.buffers = _data['buffers'].copy()
            del _data

            for epsd in tqdm(range(self.num_demo)):
                episode = {}
                episode['o'] = demoBuffer.buffers['o'][None, epsd]
                episode['ag'] = demoBuffer.buffers['ag'][None, epsd]
                episode['g'] = demoBuffer.buffers['g'][None, epsd]
                episode['u'] = demoBuffer.buffers['u'][None, epsd]

                print("Demo buffer size currently ", demoBuffer.get_current_size())

                if update_stats:
                    # add transitions to normalizer to normalize the demo data as well
                    episode['o_2'] = episode['o'][:, 1:, :]
                    episode['ag_2'] = episode['ag'][:, 1:, :]
                    num_normalizing_transitions = transitions_in_episode_batch(episode)
                    transitions = self.sample_transitions(episode, num_normalizing_transitions)

                    o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
                    transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
                    # No need to preprocess the o_2 and g_2 since this is only used for stats

                    self.o_stats.update(transitions['o'])
                    self.g_stats.update(transitions['g'])

                    self.o_stats.recompute_stats()
                    self.g_stats.recompute_stats()
                episode.clear()

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """

        self.buffer.store_episode(episode_batch)

        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
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

    def sample_batch(self):

        if self.bc_loss:
            transitions = self.buffer.sample(self.batch_size - self.demo_batch_size)
            global demoBuffer

            transitionsDemo = demoBuffer.sample(self.demo_batch_size)
            for k, values in transitionsDemo.items():
                rolloutV = transitions[k].tolist()
                for v in values:
                    rolloutV.append(v.tolist())
                transitions[k] = np.array(rolloutV)
        else:
            transitions = self.buffer.sample(self.batch_size)


        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]

        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def train(self, stage=True, policy_update=True):
        batch = self.sample_batch()
        if stage:
            self.stage_batch(batch)
        critic_1_loss, critic_2_loss, Q_grad = self._q_grads()

        if stage:
            self.stage_batch(batch)
        if policy_update:
            actor_loss, pi_grad, cloning_loss = self._pi_grads()
        else:
            actor_loss, _, cloning_loss = self._pi_grads()

        self._q_update(Q_grad)

        if policy_update:
            self._pi_update(pi_grad)

        return critic_1_loss, critic_2_loss, actor_loss, cloning_loss

    def _q_grads(self):
        # Avoid feed_dict here for performance!
        critic_1_loss, critic_2_loss, Q_grad = self.sess.run([
            self.q1_loss_tf,
            self.q2_loss_tf,
            self.Q_grad_tf,
        ])
        return critic_1_loss, critic_2_loss, Q_grad

    def _pi_grads(self):
        # Avoid feed_dict here for performance!
        cloning_loss, actor_loss, pi_grad = self.sess.run([
            self.cloning_loss_tf,
            self.pi_loss_tf,
            self.pi_grad_tf
        ])
        return actor_loss, pi_grad, cloning_loss

    def _q_update(self, Q_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)

    def _pi_update(self, pi_grad):
        self.pi_adam.update(pi_grad, self.pi_lr)

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        logger.info("Creating a TD3 agent with action space %d x %s..." % (self.dimu, self.max_u))

        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i]) for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        mask = np.concatenate((np.zeros(self.batch_size - self.demo_batch_size), np.ones(self.demo_batch_size)), axis=0)

        # Main output from computation graph
        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()

        # Target policy network
        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            # Prepare placeholder
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            # target_batch_tf['u'] = batch_tf['u']

            self.target = self.create_actor_critic(target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()

        # Target Q networks
        with tf.variable_scope('target', reuse=True) as vs:
            if reuse:
                vs.reuse_variables()
            # Target policy smoothing, by adding clipped noise to target actions
            epsilon = tf.random_normal(tf.shape(self.target.pi_tf), stddev=self.target_noise)
            epsilon = tf.clip_by_value(epsilon, -self.target_noise_clip, self.target_noise_clip)
            a2 = tf.add(self.target.pi_tf, epsilon, name='action_add_noise')
            a2 = tf.clip_by_value(a2, -self.max_u, self.max_u)

            # Prepare placeholder
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            target_batch_tf['u'] = a2

            self.target = self.create_actor_critic(target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()

        assert len(self._vars("main")) == len(self._vars("target"))

        # Bellman backup for Q functions, using Clipped Double-Q targets
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        min_q_targ = tf.minimum(self.target.q1_tf, self.target.q2_tf)
        backup = tf.stop_gradient(
            tf.clip_by_value((batch_tf['r'] + self.gamma * (1 - batch_tf['d']) * min_q_targ), *clip_range)
        )

        # TD3 loss functions
        self.q1_loss_tf = tf.reduce_mean((self.main.q1_tf - backup) ** 2)
        self.q2_loss_tf = tf.reduce_mean((self.main.q2_tf - backup) ** 2)
        self.Q_loss_tf = self.q1_loss_tf + self.q2_loss_tf

        if self.bc_loss == 1 and self.q_filter == 1:
            # where is the demonstrator action better than actor action according to the critic?
            maskMain = tf.reshape(tf.boolean_mask(self.main.q1_tf > self.main.q1_pi_tf, mask), [-1])

            self.cloning_loss_tf = tf.reduce_sum(
                (tf.boolean_mask(tf.boolean_mask(self.main.pi_tf, mask), maskMain, axis=0) -
                 tf.boolean_mask(tf.boolean_mask(batch_tf['u'], mask), maskMain, axis=0)) ** 2
            )
            self.pi_loss_tf = -self.lambda1 * tf.reduce_mean(self.main.q1_pi_tf)
            self.pi_loss_tf += self.lambda1 * self.action_l2 * tf.reduce_mean((self.main.pi_tf / self.max_u) ** 2)
            self.pi_loss_tf += self.lambda2 * self.cloning_loss_tf

        elif self.bc_loss == 1 and self.q_filter == 0:
            self.cloning_loss_tf = tf.reduce_sum((tf.boolean_mask(self.main.pi_tf, mask) -
                 tf.boolean_mask(batch_tf['u'], mask)) ** 2
            )
            self.pi_loss_tf = -self.lambda1 * tf.reduce_mean(self.main.q1_pi_tf)
            self.pi_loss_tf += self.lambda1 * self.action_l2 * tf.reduce_mean((self.main.pi_tf / self.max_u) ** 2)
            self.pi_loss_tf += self.lambda2 * self.cloning_loss_tf

        else:
            self.pi_loss_tf = -tf.reduce_mean(self.main.q1_pi_tf)
            self.pi_loss_tf += self.action_l2 * tf.reduce_mean((self.main.pi_tf / self.max_u) ** 2)
            self.cloning_loss_tf = tf.reduce_sum((tf.boolean_mask(self.main.pi_tf, mask) -
                                                  tf.boolean_mask(batch_tf['u'], mask)) ** 2)

        # Separate train ops for pi, q
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/q1') + self._vars('main/q2'))

        assert len(self._vars('main/q1')) + len(self._vars('main/q2')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)

        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/q1') + self._vars('main/q2'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/q1') + self._vars('main/q2'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # polyak averaging
        self.main_vars = self._vars('main/q1') + self._vars('main/q2') + self._vars('main/pi')
        self.target_vars = self._vars('target/q1') + self._vars('target/q2') + self._vars('target/pi')
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

        # Add ops to save and restore all the variables.
        # train_writer = tf.summary.FileWriter('./model_test')
        # train_writer.add_graph(self.sess.graph)

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
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

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)