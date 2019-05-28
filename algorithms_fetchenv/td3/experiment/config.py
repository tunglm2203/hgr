import numpy as np
import gym

import sys
sys.path.append('../')

from baselines import logger
from td3 import TD3
from her import make_sample_her_transitions


DEFAULT_ENV_PARAMS = {
    'FetchReach-v1': {
        'n_cycles': 10,
    },
}


DEFAULT_PARAMS = {
    # Scope for training (log path, n epochs, etc.)
    'root_savedir': '../../logs',
    'scope': 'td3_her',  # can be tweaked for testing
    'env_name': 'FetchPickAndPlace-v1',
    'logdir': 'use_bc_no_qfil_polyak_0.95_delayupdate_4_bs_256_tarnoise_0.2_starttraining_0_modify_update_targ',
    'num_cpu': 19,
    'seed': 0,
    'policy_save_interval': 10,
    'clip_return': 1.,
    'demo_file': '../data_generation/demonstration_FetchPickAndPlace_100_best.npz',

    'start_train_eps': 0,  # Number of steps to start using policy
    'policy_delay': 4,  # Delay policy update
    'n_epochs': int(80 * 50),
    'n_cycles': 50,  # per epoch
    'rollout_batch_size': 2,  # per mpi thread
    'n_batches': 50,  # training batches per cycle
    'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'demo_batch_size': 128,
    'prm_loss_weight': 0.001,  # Weight corresponding to the primary loss
    'aux_loss_weight': 0.0078,  # Weight corresponding to the auxilliary loss also called the cloning loss

    # For exploration
    'random_eps': 0.2,  # percentage of time a random action is taken
    'noise_eps': 0.1,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    'target_noise': 0.2,    # Std noise add to target's action for smoothing target policy
    'target_noise_clip': 0.5,      # Clipping noise for target's action

    'bc_loss': 1,  # whether or not to use the behavior cloning loss as an auxilliary loss
    'q_filter': 0,  # whether or not a Q value filter should be used on the Actor outputs
    'num_demo': 100,  # number of expert demo episodes

    # Env information
    'max_u': 1.,  # max absolute value of actions on different coordinates
    # Configure network architecture
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'actor_critic:ActorCritic',

    #
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E6),  # for experience replay
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'polyak': 0.95,  # polyak averaging coefficient

    'clip_obs': 200.,
    'relative_goals': False,

    # For evaluation
    'n_test_rollouts': 10,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network

    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future

    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
}


CACHED_ENVS = {}


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


def prepare_params(kwargs):
    # TD3 params
    td3_params = dict()

    env_name = kwargs['env_name']

    def make_env():
        return gym.make(env_name)
    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    assert hasattr(tmp_env, '_max_episode_steps')
    kwargs['T'] = tmp_env._max_episode_steps
    tmp_env.reset()
    kwargs['max_u'] = np.array(kwargs['max_u']) if isinstance(kwargs['max_u'], list) else kwargs['max_u']
    kwargs['gamma'] = 1. - 1. / kwargs['T']
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
    for name in ['buffer_size', 'hidden', 'layers',
                 'network_class',
                 'polyak',
                 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u',
                 'action_l2', 'clip_obs', 'scope', 'relative_goals',
                 'target_noise', 'target_noise_clip', 'prm_loss_weight', 'aux_loss_weight',
                 'demo_batch_size']:
        td3_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]
    kwargs['td3_params'] = td3_params
    return kwargs


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def configure_her(params):
    env = cached_make_env(params['make_env'])
    env.reset()

    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    # Prepare configuration for HER.
    her_params = {
        'reward_fun': reward_fun,
    }
    for name in ['replay_strategy', 'replay_k']:
        her_params[name] = params[name]
        params['_' + name] = her_params[name]
        del params[name]
    sample_her_transitions = make_sample_her_transitions(**her_params)

    return sample_her_transitions


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b


def configure_td3(dims, params, reuse=False, use_mpi=True, clip_return=True):
    sample_her_transitions = configure_her(params)
    # Extract relevant parameters.
    gamma = params['gamma']
    rollout_batch_size = params['rollout_batch_size']
    td3_params = params['td3_params']

    input_dims = dims.copy()

    # TD3 agent
    env = cached_make_env(params['make_env'])
    env.reset()
    td3_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'T': params['T'],
                        'clip_pos_returns': True,  # clip positive returns
                        'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # max abs of return
                        'rollout_batch_size': rollout_batch_size,
                        'subtract_goals': simple_goal_subtract,
                        'sample_transitions': sample_her_transitions,
                        'gamma': gamma,
                        'bc_loss': params['bc_loss'],
                        'q_filter': params['q_filter'],
                        'num_demo': params['num_demo'],
                        })
    td3_params['info'] = {
        'env_name': params['env_name'],
    }
    policy = TD3(reuse=reuse, **td3_params, use_mpi=use_mpi)
    return policy


def configure_dims(params):
    env = cached_make_env(params['make_env'])
    env.reset()
    obs, _, done, info = env.step(env.action_space.sample())

    dims = {
        'o': obs['observation'].shape[0],
        'u': env.action_space.shape[0],
        'g': obs['desired_goal'].shape[0],
        'd': np.array([done]).shape[0]
    }

    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims['info_{}'.format(key)] = value.shape[0]
    return dims