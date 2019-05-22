import os
import sys

import argparse
import numpy as np
import json
from mpi4py import MPI

from baselines import logger
from baselines.common import set_global_seeds, tf_util
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker

from subprocess import CalledProcessError
from baselines.her.util import mpi_fork
import tensorflow as tf
from tqdm import tqdm
from tensorboardX import SummaryWriter
from baselines.her.my_utils import tensorboard_log
import multiprocessing, logging

# os.environ['CUDA_VISIBLE_DEVICES']='0,1'


def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(policy, rollout_worker, evaluator, n_epochs, n_test_rollouts, n_cycles, n_batches,
          policy_save_interval, demo_file, logdir, use_per, log_interval=-1):
    if log_interval == -1:
        log_interval = n_cycles - 1

    rank = MPI.COMM_WORLD.Get_rank()

    tensorboard = SummaryWriter(logdir)
    latest_policy_path = os.path.join(logdir, 'policy_latest.pkl')
    best_policy_path = os.path.join(logdir, 'policy_best.pkl')
    periodic_policy_path = os.path.join(logdir, 'policy_epoch_{}_cycle_{}.pkl')

    best_success_rate = -1
    cloning_loss = 0.
    time_step = 0

    if policy.bc_loss == 1:
        logger.info("\n[INFO] Initializing demonstration buffer...")
        policy.init_demo_buffer(demo_file)  # initialize demo buffer if training with demonstrations

    # num_timesteps = n_epochs * n_cycles * rollout_length * number of rollout workers
    logger.info("\n[INFO] Start training...")
    for epoch in tqdm(range(n_epochs)):
        logger.info("\n[INFO] %s" % logdir)
        rollout_worker.clear_history()
        total_q1_loss, total_pi_loss, total_cloning_loss = 0.0, 0.0, 0.0
        for cycle_idx in tqdm(range(n_cycles)):
            episode = rollout_worker.generate_rollouts()
            time_step += rollout_worker.time_horizon * rollout_worker.rollout_batch_size
            policy.store_episode(episode)
            # cpname = multiprocessing.current_process().name
            # mpl.info("\n{}-rank{} size buffer: {}".format(cpname, rank, policy.buffer.get_current_size()))

            for _ in range(n_batches):
                if policy.bc_loss:
                    td_errors, critic_loss, actor_loss, cloning_loss = policy.train(time_step=time_step)
                else:
                    td_errors, critic_loss, actor_loss = policy.train(time_step=time_step)

                if use_per:
                    if policy.bc_loss:
                        new_priorities = np.abs(td_errors[:policy.batch_size - policy.demo_batch_size]) + \
                                         policy.prioritized_replay_eps
                        new_priorities_for_bc = \
                            np.abs(td_errors[policy.demo_batch_size:]) + policy.prioritized_replay_eps
                        policy.buffer.update_priorities(policy.episode_idxs, new_priorities, policy.transition_idxs)
                        policy.demo_buffer.update_priorities(policy.episode_idxs_for_bc, new_priorities_for_bc,
                                                             policy.transition_idxs_for_bc)
                    else:
                        new_priorities = np.abs(td_errors) + policy.prioritized_replay_eps
                        policy.buffer.update_priorities(policy.episode_idxs, new_priorities, policy.transition_idxs)

                total_q1_loss += critic_loss
                total_pi_loss += actor_loss
                total_cloning_loss += cloning_loss if policy.bc_loss else 0.

            policy.update_target_net()

            log_dict = {}
            if (cycle_idx != 0 or epoch == 0) and cycle_idx % log_interval == 0:
                evaluator.clear_history()
                for _ in range(n_test_rollouts):
                    evaluator.generate_rollouts()

                logger.record_tabular('time_step', policy.buffer.n_transitions_stored)   # record logs
                for key, val in evaluator.logs('test'):
                    logger.record_tabular(key, mpi_average(val))
                    log_dict[key] = mpi_average(val)
                for key, val in rollout_worker.logs('train'):
                    logger.record_tabular(key, mpi_average(val))
                    log_dict[key] = mpi_average(val)
                for key, val in policy.logs():
                    logger.record_tabular(key, mpi_average(val))
                    log_dict[key] = mpi_average(val)

                log_dict['q1_loss'] = total_q1_loss / n_batches
                log_dict['pi_loss'] = total_pi_loss / n_batches
                log_dict['max episode priority'] = policy.buffer._max_episode_priority
                log_dict['max transition priority'] = policy.buffer._max_transition_priority
                if policy.bc_loss:
                    log_dict['cloning_loss'] = total_cloning_loss / n_batches
                if rank == 0:
                    logger.dump_tabular()

                # save the policy if it's better than the previous ones
                success_rate = mpi_average(evaluator.current_success_rate())
                if rank == 0 and success_rate >= best_success_rate and logdir:
                    best_success_rate = success_rate
                    logger.info('New best success rate: {}. Saving to {} ...'.format(best_success_rate,
                                                                                     best_policy_path))
                    evaluator.save_policy(best_policy_path)

                if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and logdir:
                    logger.info('Saving periodic policy to {}...'.format(periodic_policy_path.format(epoch, cycle_idx)))
                    evaluator.save_policy(latest_policy_path)
                    evaluator.save_policy(periodic_policy_path.format(epoch, cycle_idx))

                # make sure that different threads have different seeds
                local_uniform = np.random.uniform(size=(1,))
                root_uniform = local_uniform.copy()
                MPI.COMM_WORLD.Bcast(root_uniform, root=0)
                if rank != 0:
                    assert local_uniform[0] != root_uniform[0]
                tensorboard_log(tensorboard, log_dict, policy.buffer.n_transitions_stored)
        policy.save(os.path.join(logdir, rollout_worker.envs[0].spec.id.split('-')[0].lower() + '_model.gz'))
    return policy


def launch(env, logdir, n_epochs, num_cpu, seed=None,
           replay_strategy='future', policy_save_interval=1, clip_return=True,
           demo_file=None, override_params=None, load_path=None, log_interval=-1):
    """
    launch training with mpi
    :param env: (str) environment ID
    :param logdir: (str) the log directory
    :param n_epochs: (int) the number of training epochs
    :param num_cpu: (int) the number of CPUs to run on
    :param seed: (int) the initial random seed
    :param replay_strategy: (str) the type of replay strategy ('future' or 'none')
    :param policy_save_interval: (int) the interval with which policy pickles are saved.
        If set to 0, only the best and latest policy will be pickled.
    :param clip_return: (float): clip returns to be in [-clip_return, clip_return]
    :param demo_file: (str) the path to the demonstration file
    :param override_params: (dict) override any parameter for training
    :param load_path: (str) the path to pretrained model (path/to/pretrained.gz)
    :param log_interval: (int) Log interval, if equal to -1, saving at end of epoch
    """
    if override_params is None:
        override_params = {}

    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        try:
            whoami = mpi_fork(num_cpu, ['--bind-to', 'core'])
        except CalledProcessError:
            # fancy version of mpi call failed, try simple version
            whoami = mpi_fork(num_cpu)

        if whoami == 'parent':
            sys.exit(0)
        tf_util.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(folder=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank if seed is not None else None
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['env_name'] = env
    params['replay_strategy'] = replay_strategy
    if env in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env])  # merge env-specific parameters in

    params.update(**override_params)  # makes it possible to override any parameter

    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_params(params)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    dims = config.configure_dims(params)
    total_timesteps = n_epochs * params['n_cycles'] * params['time_horizon'] * params['rollout_batch_size']
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return, total_timesteps=total_timesteps)

    # Add ops to save and restore all the variables.
    save_graph = tf.summary.FileWriter(logdir)
    save_graph.add_graph(policy.sess.graph)

    if load_path != '':
        tf_util.load_variables(load_path, sess=policy.sess)
        params['load_path'] = load_path

    # Overwriting variables which are overwrited in DDPG class, command line, etc.
    params['_buffer_size'] = policy.buffer_size
    params['_num_cpu'] = num_cpu
    params['_log_interval'] = log_interval
    params['_logdir'] = logdir
    params['_demo_file'] = demo_file
    config.log_params(params, logger_input=logger)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'compute_q': False,
        'time_horizon': params['time_horizon'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'compute_q': True,
        'time_horizon': params['time_horizon'],
    }

    for name in ['time_horizon', 'rollout_batch_size', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    train(policy=policy, rollout_worker=rollout_worker, evaluator=evaluator,
          n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'], n_cycles=params['n_cycles'],
          n_batches=params['n_batches'], policy_save_interval=policy_save_interval,
          demo_file=demo_file, logdir=logdir, use_per=params['_use_per'],
          log_interval=log_interval)


parser = argparse.ArgumentParser(description='Reinforcement Learning')
parser.add_argument('--env', type=str, default='FetchPickAndPlace-v1', help='The name of the Gym environment')
parser.add_argument('--logdir', type=str, default='logs/pickplace/per/profile', help='Path to save training '
                                                                                     'information (model, log, etc.)')
parser.add_argument('--n_epochs', type=int, default=125, help='The number of training epochs to run')
parser.add_argument('--num_cpu', type=int, default=1, help='The number of CPU cores to use (using MPI)')
parser.add_argument('--seed', type=int, default=0, help='Seed both the environment and the training code')
parser.add_argument('--policy_save_interval', type=int, default=1, help='If 0 only the best & latest policy be pickled')
parser.add_argument('--log_interval', type=int, default=-1, help='-1 is printing at end of epoch')
parser.add_argument('--replay_strategy', type=str, default='future', help='"future" uses HER, "none" disables HER')
parser.add_argument('--clip_return', action='store_false', help='Whether or not returns should be clipped')
parser.add_argument('--demo_file', type=str, default='experiment/data_generation/demonstration_FetchPickAndPlace.npz',
                    help='Demo data file path')
parser.add_argument('--load_path', type=str, default='', help='Pretrained model path')
# Get argument from command line
args = parser.parse_args()
print('--------- YOUR SETTING ---------')
for arg in vars(args):
    print("%15s: %s" % (str(arg), str(getattr(args, arg))))
print("")


def main():
    launch(env=args.env,
           logdir=args.logdir,
           n_epochs=args.n_epochs,
           num_cpu=args.num_cpu,
           seed=args.seed,
           replay_strategy=args.replay_strategy,
           policy_save_interval=args.policy_save_interval,
           clip_return=args.clip_return,
           demo_file=args.demo_file,
           override_params=None,
           load_path=args.load_path,
           log_interval=args.log_interval)


if __name__ == '__main__':
    main()
