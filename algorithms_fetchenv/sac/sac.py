import sys
import time
import multiprocessing
from collections import deque
import warnings

import numpy as np
import tensorflow as tf

from stable_baselines.a2c.utils import find_trainable_variables, total_episode_reward_logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from my_replay_buffer import HindsightExperientReplay
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from policies import SACPolicy
from stable_baselines import logger
sys.path.append('../')
from utils import unpack_obs, compute_success_rate_from_list, compute_success_rate, my_tensorboard_logger
import os
from tqdm import tqdm
import pickle


def get_vars(scope):
    """
    Alias for get_trainable_vars

    :param scope: (str)
    :return: [tf Variable]
    """
    return tf_util.get_trainable_vars(scope)


class SAC(OffPolicyRLModel):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup) and from the Softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    :param policy: (SACPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update", between 0 and 1)
    :param ent_coef: (str or float) Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_update_interval: (int) update the target network every `target_network_update_freq` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param target_entropy: (str or float) target entropy when learning ent_coef (ent_coef = 'auto')
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        Note: this has no effect on SAC logging for now
    """

    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 learning_starts=100, train_freq=1, batch_size=64,
                 tau=0.005, ent_coef='auto', target_update_interval=1,
                 gradient_steps=1, target_entropy='auto', verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 use_bc=False, use_q_filter=False,
                 demo_batchsize=0, demo_file='', load_from_pickle=False, pickle_file=''):

        super(SAC, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                  policy_base=SACPolicy, requires_vec_env=False, policy_kwargs=policy_kwargs)

        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau
        # In the original paper, same learning rate is used for all networks
        # self.policy_lr = learning_rate
        # self.qf_lr = learning_rate
        # self.vf_lr = learning_rate
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.gamma = gamma

        self.value_fn = None
        self.graph = None
        self.replay_buffer = None
        self.episode_reward = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.target_entropy = target_entropy
        self.full_tensorboard_log = full_tensorboard_log

        self.obs_target = None
        self.target_policy = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.value_target = None
        self.step_ops = None
        self.target_update_op = None
        self.infos_names = None
        self.entropy = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.log_ent_coef = None

        self.use_bc = use_bc
        self.use_q_filter = use_q_filter
        if use_bc:
            self.load_from_pickle = load_from_pickle
            self.pickle_file = pickle_file
            self.bc_coef = 0.001
            self.demo_buffer = None
            self.cloning_loss = None
            self.demo_batchsize = demo_batchsize
            self._init_demonstration(demo_file, load_from_pickle, pickle_file)

        if _init_setup_model:
            self.setup_model()

        self.env = gym.make(self.env.spec.id)

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        deterministic_action = self.deterministic_action * np.abs(self.action_space.low)
        return policy.obs_ph, self.actions_ph, deterministic_action

    def _init_demonstration(self, demo_file, from_obj_pickle=False, pickle_file=''):
        """
        Initialize demonstration buffer
        NOTE: Only using for Fetch Environment
        :param demo_file: Path to the demonstration file
        """

        print('[AIM-NOTIFY] Initializing demonstration buffer...')
        # dict_key of demo_data: ['ep_rets', 'obs', 'rews', 'acs', 'info']
        if not os.path.exists(demo_file):
            print('[AIM-ERROR] The demonstration file is not exist.')
            exit()

        demo_data = np.load(demo_file)
        assert demo_data['ep_rets'].shape[0] == demo_data['obs'].shape[0] == \
               demo_data['rews'].shape[0] == demo_data['acs'].shape[0] == demo_data['info'].shape[0], \
            print('[AIM-ERROR] The number of demonstrations is not equal in each field.')

        assert demo_data['acs'].shape[0] == demo_data['rews'].shape[0] == demo_data['info'].shape[0],\
            print('[AIM-ERROR] The number of transitions (acs, rews, info) in each episode not equal.')

        n_demos = demo_data['ep_rets'].shape[0]
        n_transition_per_episode = demo_data['rews'].shape[1]

        assert demo_data['obs'].shape[1] == n_transition_per_episode + 1, \
            print('[AIM-ERROR] The observation in demonstration not including terminal state.')

        demo_buff_size = n_demos * n_transition_per_episode
        self.demo_buffer = HindsightExperientReplay(demo_buff_size, env.spec.id)

        if from_obj_pickle:
            pickle_in = open(pickle_file, "rb")
            _data = pickle.load(pickle_in)
            self.demo_buffer._storage = _data.copy()
            self.demo_buffer.current_n_episodes = len(_data['u'])
            del _data
        else:
            # Read and store all demonstrations into self.demo_buffer
            for ep_idx in tqdm(range(n_demos)):
                # Store from `start state` to `terminal state - 1`
                for t in range(n_transition_per_episode):
                    if t < n_transition_per_episode - 1:
                        self.demo_buffer.add(demo_data['obs'][ep_idx][t]['observation'],
                                             demo_data['obs'][ep_idx][t]['desired_goal'],
                                             demo_data['obs'][ep_idx][t]['achieved_goal'],
                                             np.array([1.0, 1.0, 1.0, 1.0]), # demo_data['acs'][ep_idx][t]
                                             False,
                                             demo_data['info'][ep_idx][t])
                    else:
                        self.demo_buffer.add(demo_data['obs'][ep_idx][t]['observation'],
                                             demo_data['obs'][ep_idx][t]['desired_goal'],
                                             demo_data['obs'][ep_idx][t]['achieved_goal'],
                                             np.array([1.0, 1.0, 1.0, 1.0]), #demo_data['acs'][ep_idx][t]
                                             True,
                                             demo_data['info'][ep_idx][t])
                # Push `last state` into demo_buffer
                self.demo_buffer.add(demo_data['obs'][ep_idx][t + 1]['observation'],
                                     -1, -1,
                                     demo_data['obs'][ep_idx][t + 1]['achieved_goal'],
                                     -1, -1)
            with open('demonstration_buffer.pkl', 'wb') as f:
                pickle.dump(self.demo_buffer._storage, f)

        print('[AIM-NOTIFY] Initialize demonstration buffer done.')

    def setup_model(self):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                n_cpu = multiprocessing.cpu_count()
                if sys.platform == 'darwin':
                    n_cpu //= 2
                self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

                # tung: related to replay buffer
                self.replay_buffer = HindsightExperientReplay(self.buffer_size, env.spec.id)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space,
                                                     **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy.obs_ph
                    self.processed_next_obs_ph = self.target_policy.processed_obs
                    self.action_target = self.target_policy.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    # first return value corresponds to deterministic actions
                    # policy_out corresponds to stochastic actions, used for training
                    # logp_pi is the log probabilty of actions taken by the policy
                    self.deterministic_action, policy_out, logp_pi = self.policy_tf.make_actor(self.processed_obs_ph)
                    # Monitor the entropy of the policy,
                    # this is not used for training
                    self.entropy = tf.reduce_mean(self.policy_tf.entropy)
                    #  Use two Q-functions to improve performance by reducing overestimation bias.
                    qf1, qf2, value_fn = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph,
                                                                     create_qf=True, create_vf=True)
                    qf1_pi, qf2_pi, _ = self.policy_tf.make_critics(self.processed_obs_ph, policy_out,
                                                                    create_qf=True, create_vf=False,
                                                                    reuse=True)

                    # Target entropy is used when learning the entropy coefficient
                    if self.target_entropy == 'auto':
                        # automatically set target entropy if needed
                        self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
                    else:
                        # Force conversion
                        # this will also throw an error for unexpected string
                        self.target_entropy = float(self.target_entropy)

                    # The entropy coefficient or entropy can be learned automatically
                    # see Automating Entropy Adjustment for Maximum Entropy RL section
                    # of https://arxiv.org/abs/1812.05905
                    if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
                        # Default initial value of ent_coef when learned
                        init_value = 1.0
                        if '_' in self.ent_coef:
                            init_value = float(self.ent_coef.split('_')[1])
                            assert init_value > 0., "The initial value of ent_coef must be greater than 0"

                        self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                            initializer=np.log(init_value).astype(np.float32))
                        self.ent_coef = tf.exp(self.log_ent_coef)
                    else:
                        # Force conversion to float
                        # this will throw an error if a malformed string (different from 'auto')
                        # is passed
                        self.ent_coef = float(self.ent_coef)

                with tf.variable_scope("target", reuse=False):
                    # Create the value network
                    _, _, value_target = self.target_policy.make_critics(self.processed_next_obs_ph,
                                                                         create_qf=False, create_vf=True)
                    self.value_target = value_target

                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two Q-Values (Double-Q Learning)
                    min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                    # Targets for Q and V regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * self.value_target
                    )

                    # Compute Q-Function loss
                    # TODO: test with huber loss (it would avoid too high values)
                    qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
                    qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

                    # Compute the entropy temperature loss
                    # it is used when the entropy coefficient is learned
                    ent_coef_loss, entropy_optimizer = None, None
                    if not isinstance(self.ent_coef, float):
                        ent_coef_loss = -tf.reduce_mean(
                            self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                        entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

                    # Compute the policy loss
                    # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                    if self.use_bc:
                        mask = np.concatenate(
                            (np.zeros(self.batch_size - self.demo_batchsize), np.ones(self.demo_batchsize)), axis=0)
                        if self.use_q_filter:
                            mask_q_filter = tf.reshape(tf.boolean_mask(qf1 > qf1_pi, mask), [-1])    #tung: should use qf1 > qf1_pi or min(qf1, qf2) > qf1_pi ?

                            self.cloning_loss = tf.reduce_sum(
                                tf.square(
                                    tf.boolean_mask(tf.boolean_mask(self.deterministic_action, mask), mask_q_filter, axis=0) -
                                    tf.boolean_mask(tf.boolean_mask(self.actions_ph, mask), mask_q_filter, axis=0)
                                )
                            )
                            policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi -
                                                            qf1_pi +
                                                            # tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u)) +
                                                            self.bc_coef * self.cloning_loss)
                        else:
                            self.cloning_loss = tf.reduce_sum(
                                tf.square(
                                    tf.boolean_mask(self.deterministic_action, mask) -
                                    tf.boolean_mask(self.actions_ph, mask))
                            )
                            policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi - qf1_pi +
                                                            self.bc_coef * self.cloning_loss)
                        # policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi - qf1_pi)
                    else:
                        policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi - qf1_pi)

                    # NOTE: in the original implementation, they have an additional
                    # regularization loss for the gaussian parameters
                    # this is not used for now
                    # policy_loss = (policy_kl_loss + policy_regularization_loss)
                    policy_loss = policy_kl_loss

                    # We update the vf towards the min of two Q-functions in order to
                    # reduce overestimation bias from function approximation error.
                    v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * logp_pi)
                    value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)

                    values_losses = qf1_loss + qf2_loss + value_loss

                    # Policy train op
                    # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    policy_train_op = policy_optimizer.minimize(policy_loss, var_list=get_vars('model/pi'))

                    # Value train op
                    value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    values_params = get_vars('model/values_fn')

                    source_params = get_vars("model/values_fn/vf")
                    target_params = get_vars("target/values_fn/vf")

                    # Polyak averaging for target variables
                    self.target_update_op = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]
                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                    # and we first need to compute the policy action before computing q values losses
                    with tf.control_dependencies([policy_train_op]):
                        train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                        self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'value_loss', 'entropy']
                        # All ops to call during one training step
                        self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                                         value_loss, qf1, qf2, value_fn, logp_pi,
                                         self.entropy, policy_train_op, train_values_op]

                        # Add entropy coefficient optimization operation if needed
                        if ent_coef_loss is not None:
                            with tf.control_dependencies([train_values_op]):
                                ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                                self.infos_names += ['ent_coef_loss', 'ent_coef']
                                self.step_ops += [ent_coef_op, ent_coef_loss, self.ent_coef]

                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('qf1_loss', qf1_loss)
                    tf.summary.scalar('qf2_loss', qf2_loss)
                    tf.summary.scalar('value_loss', value_loss)
                    tf.summary.scalar('entropy', self.entropy)
                    if ent_coef_loss is not None:
                        tf.summary.scalar('ent_coef_loss', ent_coef_loss)
                        tf.summary.scalar('ent_coef', self.ent_coef)

                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = find_trainable_variables("model")
                self.target_params = find_trainable_variables("target/values_fn/vf")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

    def _train_step(self, step, writer, learning_rate):
        # Sample a batch from the replay buffer
        # tung: Sample from replay buffer
        if self.use_bc:
            assert self.demo_batchsize < self.batch_size, \
                print('[AIM-NOTIFY] Batch size of demonstration buffer larger than batch size of replay buffer.')
            batch_1 = self.replay_buffer.sample(self.batch_size - self.demo_batchsize)  # Sample from replay buffer
            batch_2 = self.demo_buffer.sample(self.demo_batchsize)    # Sample from demo buffer
            batch = ()
            for i in range(len(batch_1)):
                batch += (np.concatenate((batch_1[i], batch_2[i]), axis=0), )
            del batch_1, batch_2
        else:
            batch = self.replay_buffer.sample(self.batch_size)

        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate
        }

        # out  = [policy_loss, qf1_loss, qf2_loss,
        #         value_loss, qf1, qf2, value_fn, logp_pi,
        #         self.entropy, policy_train_op, train_values_op]

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + self.step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(self.step_ops, feed_dict)

        # Unpack to monitor losses and entropy
        policy_loss, qf1_loss, qf2_loss, value_loss, *values = out
        # qf1, qf2, value_fn, logp_pi, entropy, *_ = values
        entropy = values[4]

        if self.log_ent_coef is not None:
            ent_coef_loss, ent_coef = values[-2:]
            return policy_loss, qf1_loss, qf2_loss, value_loss, entropy, ent_coef_loss, ent_coef

        return policy_loss, qf1_loss, qf2_loss, value_loss, entropy

    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=4, tb_log_name="SAC", reset_num_timesteps=True):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn(seed)

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()
            episode_rewards = [0.0]
            obs = self.env.reset()
            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            n_updates = 0
            infos_values = []

            # Variables used to log into tensorboard
            episode_idx = 0
            train_success_rate = 0.0
            test_success_rate = 0.0
            env_test = gym.make(self.env.spec.id)
            env_test = gym.wrappers.FlattenDictWrapper(env_test, ['observation', 'desired_goal'])
            n_eps_compute_reward = 100

            for step in range(total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy.
                if self.num_timesteps < self.learning_starts:
                    action = self.env.action_space.sample()
                    # No need to rescale when sampling random action
                    rescaled_action = action
                else:
                    _, _, o, _ = unpack_obs(obs)
                    action = self.policy_tf.step(o[None], deterministic=False).flatten()
                    # Rescale from [-1, 1] to the correct bounds
                    rescaled_action = action * np.abs(self.action_space.low)

                assert action.shape == self.env.action_space.shape

                new_obs, reward, done, info = self.env.step(rescaled_action)
                # Store transition in the replay buffer.
                # tung: Add to replay buffer
                self.replay_buffer.add(obs['observation'], obs['desired_goal'], obs['achieved_goal'],
                                       action, done, info)
                obs = new_obs

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                      ep_done, writer, self.num_timesteps)
                episode_rewards[-1] += reward

                if done:
                    self.replay_buffer.add(obs_t=obs['observation'],
                                           action=-1,
                                           desired_goal=-1,
                                           achieved_goal=obs['achieved_goal'],
                                           done=-1,
                                           info=-1)

                    mb_infos_vals = []

                    ep_info = self.replay_buffer.get_last_episodes(n_eps_compute_reward, 'info_is_success')
                    if ep_info is not None:
                        train_success_rate = compute_success_rate_from_list(ep_info)
                    my_tensorboard_logger("Train_success_rate", train_success_rate, writer, self.num_timesteps)

                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        if self.replay_buffer.current_n_episodes * self.replay_buffer.horizon < self.batch_size or \
                                self.num_timesteps < self.learning_starts:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.learning_rate(frac)
                        # Update policy and critics (q functions)
                        mb_infos_vals.append(self._train_step(step, writer, current_lr))

                    # Update target network
                    if episode_idx % self.target_update_interval == 0:
                        # Update target network
                        self.sess.run(self.target_update_op)
                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)

                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)
                    episode_idx += 1

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                self.num_timesteps += 1
                # Display training infos
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.logkv("Train success rate", train_success_rate)
                    # tung: modify interval for test
                    if len(episode_rewards) % n_eps_compute_reward == 0:

                        obs_test = env_test.reset()
                        infos_test, info_ep_test = [], []
                        for _ in range(n_eps_compute_reward):
                            while True:
                                action_test, _ = model.predict(obs_test)
                                obs_test, rewards_test, done_test, info_test = env_test.step(action_test)
                                info_ep_test.append(info_test)
                                if done_test:
                                    infos_test.append(info_ep_test)
                                    info_ep_test = []
                                    break
                        test_success_rate = compute_success_rate(infos_test)
                        my_tensorboard_logger("Test_success_rate", test_success_rate, writer, self.num_timesteps)
                        logger.logkv("TEST SUCCESS RATE", test_success_rate)
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []
            return self

    def action_probability(self, observation, state=None, mask=None, actions=None):
        if actions is None:
            warnings.warn("Even thought SAC has a Gaussian policy, it cannot return a distribution as it "
                          "is squashed by an tanh before being scaled and ouputed. Therefore 'action_probability' "
                          "will only work with the 'actions' keyword argument being used. Returning None.")
            return None

        observation = np.array(observation)

        warnings.warn("The probabilty of taken a given action is exactly zero for a continuous distribution."
                      "See http://blog.christianperone.com/2019/01/ for a good explanation")

        return np.zeros((observation.shape[0], 1), dtype=np.float32)

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy_tf.step(observation, deterministic=deterministic)
        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = actions * np.abs(self.action_space.low)  # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def save(self, save_path):
        data = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "ent_coef": self.ent_coef if isinstance(self.ent_coef, float) else 'auto',
            "target_entropy": self.target_entropy,
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            # "replay_buffer": self.replay_buffer
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params = self.sess.run(self.params)
        target_params = self.sess.run(self.target_params)

        self._save_to_file(save_path, data=data, params=params + target_params)

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        data, params = cls._load_from_file(load_path)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(policy=data["policy"], env=env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()

        restores = []
        for param, loaded_p in zip(model.params + model.target_params, params):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        return model


from policies import FeedForwardPolicy


class CustomSACPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                              layers=[256, 256, 256],
                                              layer_norm=False,
                                              feature_extraction="mlp")


if __name__ == '__main__':
    import gym
    import numpy as np

    from stable_baselines.common.vec_env import DummyVecEnv
    from policies import MlpPolicy
    env = gym.make('FetchPickAndPlace-v1')
    # env = gym.make('FetchReach-v1')
    env = gym.wrappers.FlattenDictWrapper(env, ['observation', 'desired_goal'])
    env.observation_space.high = 200.0
    env.observation_space.low = -200.0

    # env = DummyVecEnv([lambda: env])
    save_dir = os.path.join('logs', 'sac_her_entAuto_delay_update_target' + env.spec.id.split('-')[0])
    save_filename = os.path.join(save_dir, 'sac_' + env.spec.id.split('-')[0])

    model = SAC(CustomSACPolicy, env, verbose=1,
                learning_starts=100,
                buffer_size=int(100000), #1e6
                tensorboard_log=save_dir,
                learning_rate=1e-4,
                batch_size=256,
                tau=0.05,
                target_entropy='auto',
                ent_coef='auto',
                gradient_steps=50,
                use_bc=True,
                demo_batchsize=128,
                demo_file='../her/data_generation/demonstration_FetchPickAndPlace.npz',
                load_from_pickle=True,
                use_q_filter=False,
                pickle_file='demonstration_buffer.pkl'
                )

    model.learn(total_timesteps=100000, log_interval=2)
    model.save(save_filename)

    # del model  # remove to demonstrate saving and loading
    # model = SAC.load(save_filename, env)
    #
    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()
    #     if dones:
    #         obs = env.reset()
