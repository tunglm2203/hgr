import numpy as np
import gym

from baselines import logger
from baselines.her.ddpg import DDPG
from baselines.her.her_sampler import make_sample_her_transitions

DEFAULT_ENV_PARAMS = {
    'FetchReach-v1': {
        'n_cycles': 10,
        'n_batches': 50,
        'rollout_batch_size': 2,
        'batch_size': 256,
        'n_test_rollouts': 5,
        'random_eps': 0.3,
        'noise_eps': 0.2,

        'bc_loss': False,
        'q_filter': False,

        'use_per': True,
        'prioritized_replay_alpha': 0.6,
        'prioritized_replay_beta0': 0.4,
        'prioritized_replay_beta_iters': None,
        'prioritized_replay_alpha_prime': 0.0,
        'prioritized_replay_beta0_prime': 0.0,
        'prioritized_replay_beta_iters_prime': None,
        'prioritized_replay_eps': 1e-6,
        'use_huber_loss': False,
        'buffer_size': int(8E5),
    },
    'FetchPickAndPlace-v1': {
        'n_cycles': 40,
        'n_batches': 40,
        'rollout_batch_size': 8,
        'batch_size': 256,
        'n_test_rollouts': 5,
        'random_eps': 0.1,
        'noise_eps': 0.1,

        'bc_loss': True,
        'q_filter': True,
        'demo_batch_size': 128,
        'num_demo': 100,
        'prm_loss_weight': 0.001,
        'aux_loss_weight': 0.0078,

        'use_per': False,  # default: True
        'prioritized_replay_alpha': 0.6,
        'prioritized_replay_beta0': 0.4,
        'prioritized_replay_beta_iters': None,
        'prioritized_replay_alpha_prime': 0.0,
        'prioritized_replay_beta0_prime': 0.0,
        'prioritized_replay_beta_iters_prime': None,
        'prioritized_replay_eps': 1e-6,
        'use_huber_loss': False,
        'buffer_size': int(8E5),
    },
    'FetchPush-v1': {
        'n_cycles': 50,
        'n_batches': 40,
        'rollout_batch_size': 2,
        'batch_size': 256,
        'n_test_rollouts': 5,
        'random_eps': 0.3,
        'noise_eps': 0.2,

        'bc_loss': False,
        'q_filter': False,
        'demo_batch_size': 128,
        'num_demo': 100,
        'prm_loss_weight': 0.001,
        'aux_loss_weight': 0.0078,

        'use_per': True,
        'prioritized_replay_alpha': 0.6,
        'prioritized_replay_beta0': 0.4,
        'prioritized_replay_beta_iters': None,
        'prioritized_replay_alpha_prime': 0.6,
        'prioritized_replay_beta0_prime': 0.4,
        'prioritized_replay_beta_iters_prime': None,
        'prioritized_replay_eps': 1e-6,
        'use_huber_loss': False,
        'buffer_size': int(1E6),
    },
    'FetchSlide-v1': {
        'n_cycles': 50,
        'n_batches': 40,
        'rollout_batch_size': 2,
        'batch_size': 256,
        'n_test_rollouts': 5,
        'random_eps': 0.3,
        'noise_eps': 0.2,

        'bc_loss': False,
        'q_filter': False,
        'demo_batch_size': 128,
        'num_demo': 100,
        'prm_loss_weight': 0.001,
        'aux_loss_weight': 0.0078,

        'use_per': False,
        'prioritized_replay_alpha': 0.6,
        'prioritized_replay_beta0': 0.4,
        'prioritized_replay_beta_iters': None,
        'prioritized_replay_alpha_prime': 0.7,
        'prioritized_replay_beta0_prime': 0.5,
        'prioritized_replay_beta_iters_prime': None,
        'prioritized_replay_eps': 1e-6,
        'use_huber_loss': False,
        'buffer_size': int(1E6),
    },
}

DEFAULT_PARAMS = {
    # env
    'max_u': 1.,  # max absolute value of actions on different coordinates
    # ddpg
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'baselines.her.actor_critic:ActorCritic',
    'q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E6),  # for experience replay
    'polyak': 0.95,  # polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'ddpg',  # can be tweaked for testing
    'relative_goals': False,
    # training
    'n_cycles': 50,  # per epoch
    'rollout_batch_size': 2,  # per mpi thread (number of collected episodes each time execution)
    'n_batches': 40,  # training batches per cycle
    'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 10,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration
    'random_eps': 0.3,  # percentage of time a random action is taken
    'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values

    'bc_loss': False,  # whether or not to use the behavior cloning loss as an auxilliary loss
    'q_filter': False,  # whether or not a Q value filter should be used on the Actor outputs
    'num_demo': 100,  # number of expert demo episodes
    'demo_batch_size': 128,
    # number of samples to be used from the demonstrations buffer, per mpi thread 128/1024 or 32/256
    'prm_loss_weight': 0.001,  # Weight corresponding to the primary loss
    'aux_loss_weight': 0.0078,  # Weight corresponding to the auxilliary loss also called the cloning loss

    'use_per': False,
    'prioritized_replay_alpha': 0.6,
    'prioritized_replay_beta0': 0.4,
    'prioritized_replay_beta_iters': None,
    'prioritized_replay_eps': 1e-6,

    'use_huber_loss': False,
}

CACHED_ENVS = {}


def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    :param make_env: (function (): Gym Environment) creates the environment
    :return: (Gym Environment) the created environment
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]


def prepare_params(kwargs):
    """
    prepares DDPG params from kwargs
    :param kwargs: (dict) the input kwargs
    :return: (dict) DDPG parameters
    """
    # DDPG params
    ddpg_params = dict()

    env_name = kwargs['env_name']

    def make_env():
        return gym.make(env_name)

    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    kwargs['time_horizon'] = tmp_env.spec.max_episode_steps  # wrapped envs preserve their spec

    kwargs['max_u'] = np.array(kwargs['max_u']) if isinstance(kwargs['max_u'], list) else kwargs['max_u']
    kwargs['gamma'] = 1. - 1. / kwargs['time_horizon']
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['q_lr'] = kwargs['lr']
        del kwargs['lr']
    for name in ['buffer_size', 'hidden', 'layers',
                 'network_class',
                 'polyak',
                 'batch_size', 'q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u',
                 'action_l2', 'clip_obs', 'scope', 'relative_goals',
                 'bc_loss', 'q_filter', 'num_demo', 'demo_batch_size',
                 'prm_loss_weight', 'aux_loss_weight',
                 'gamma',
                 'use_per', 'prioritized_replay_alpha', 'prioritized_replay_beta0',
                 'prioritized_replay_beta_iters', 'prioritized_replay_eps',
                 'prioritized_replay_alpha_prime', 'prioritized_replay_beta0_prime',
                 'prioritized_replay_beta_iters_prime',
                 'use_huber_loss']:
        ddpg_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]
    kwargs['ddpg_params'] = ddpg_params

    return kwargs


def log_params(params, logger_input=logger):
    """
        log the parameters
        :param params: (dict) parameters to log
        :param logger_input: (logger) the logger
    """
    logger_input.info('\nParamerter (param with prefix  `_` is system param, without `_` is DDPG param)\n')
    for key in sorted(params.keys()):
        logger_input.info('{}: {}'.format(key, params[key]))


def configure_her(params):
    """
    configure hindsight experience replay
    :param params: (dict) input parameters
    :return: (function (dict, int): dict) returns a HER update function for replay buffer batch
    """

    env = cached_make_env(params['make_env'])
    env.reset()

    def reward_fun(achieved_goal, desired_goal, info):  # vectorized
        return env.compute_reward(achieved_goal=achieved_goal, desired_goal=desired_goal, info=info)

    # Prepare configuration for HER
    her_params = {
        'reward_fun': reward_fun,
    }
    for name in ['replay_strategy', 'replay_k']:
        her_params[name] = params[name]
        params['_' + name] = her_params[name]
        del params[name]

    if params['_use_per']:
        sample_her_transitions = her_params
    else:
        sample_her_transitions = make_sample_her_transitions(**her_params)

    return sample_her_transitions


def simple_goal_subtract(vec_a, vec_b):
    """
    checks if a and b have the same shape, and does a - b
    :param vec_a: (np.ndarray)
    :param vec_b: (np.ndarray)
    :return: (np.ndarray) a - b
    """
    assert vec_a.shape == vec_b.shape
    return vec_a - vec_b


def configure_ddpg(dims, params, total_timesteps, reuse=False, clip_return=True):
    """
    configure a DDPG model from parameters
    :param dims: ({str: int}) the dimensions
    :param params: (dict) the DDPG parameters
    :param total_timesteps: (int) Number of time step for training, measure in transitions
    :param reuse: (bool) whether or not the networks should be reused
    :param clip_return: (float) clip returns to be in [-clip_return, clip_return]
    :return: (her.DDPG) the ddpg model
    """
    sample_her_transitions = configure_her(params)
    # Extract relevant parameters.
    gamma = params['_gamma']
    ddpg_params = params['ddpg_params']

    input_dims = dims.copy()

    # DDPG agent
    env = cached_make_env(params['make_env'])
    env.reset()
    ddpg_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'time_horizon': params['time_horizon'],
                        'clip_pos_returns': True,  # clip positive returns
                        'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # max abs of return
                        'subtract_goals': simple_goal_subtract,
                        'sample_transitions': sample_her_transitions,

                        'total_timesteps': total_timesteps,
                        'rollout_batch_size': params['rollout_batch_size'],
                        })
    ddpg_params['info'] = {
        'env_name': params['env_name'],
    }
    policy = DDPG(input_dims=ddpg_params['input_dims'], buffer_size=ddpg_params['buffer_size'],
                  hidden=ddpg_params['hidden'], layers=ddpg_params['layers'],
                  network_class=ddpg_params['network_class'], polyak=ddpg_params['polyak'],
                  batch_size=ddpg_params['batch_size'], q_lr=ddpg_params['q_lr'],
                  pi_lr=ddpg_params['pi_lr'], norm_eps=ddpg_params['norm_eps'],
                  norm_clip=ddpg_params['norm_clip'], max_u=ddpg_params['max_u'],
                  action_l2=ddpg_params['action_l2'], clip_obs=ddpg_params['clip_obs'], scope=ddpg_params['scope'],
                  time_horizon=ddpg_params['time_horizon'], rollout_batch_size=ddpg_params['rollout_batch_size'],
                  subtract_goals=ddpg_params['subtract_goals'], relative_goals=ddpg_params['relative_goals'],
                  clip_pos_returns=ddpg_params['clip_pos_returns'], clip_return=ddpg_params['clip_return'],
                  bc_loss=ddpg_params['bc_loss'], q_filter=ddpg_params['q_filter'], num_demo=ddpg_params['num_demo'],
                  demo_batch_size=ddpg_params['demo_batch_size'], prm_loss_weight=ddpg_params['prm_loss_weight'],
                  aux_loss_weight=ddpg_params['aux_loss_weight'], sample_transitions=ddpg_params['sample_transitions'],
                  gamma=ddpg_params['gamma'], use_per=ddpg_params['use_per'],
                  total_timesteps=ddpg_params['total_timesteps'],
                  prioritized_replay_alpha=ddpg_params['prioritized_replay_alpha'],
                  prioritized_replay_beta0=ddpg_params['prioritized_replay_beta0'],
                  prioritized_replay_beta_iters=ddpg_params['prioritized_replay_beta_iters'],
                  prioritized_replay_alpha_prime=ddpg_params['prioritized_replay_alpha_prime'],
                  prioritized_replay_beta0_prime=ddpg_params['prioritized_replay_beta0_prime'],
                  prioritized_replay_beta_iters_prime=ddpg_params['prioritized_replay_beta_iters'],
                  prioritized_replay_eps=ddpg_params['prioritized_replay_eps'],
                  use_huber_loss=ddpg_params['use_huber_loss'],
                  reuse=reuse)
    return policy


def configure_dims(params):
    env = cached_make_env(params['make_env'])
    env.reset()
    obs, _, _, info = env.step(env.action_space.sample())

    dims = {
        'o': obs['observation'].shape[0],
        'u': env.action_space.shape[0],
        'g': obs['desired_goal'].shape[0],
    }
    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims['info_{}'.format(key)] = value.shape[0]
    return dims
