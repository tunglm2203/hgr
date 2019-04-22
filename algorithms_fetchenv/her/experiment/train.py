import os
import sys
import click
import numpy as np
import json
from mpi4py import MPI
import resource

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import config
from rollout import RolloutWorker, RolloutWorkerOriginal
from util import mpi_fork

from subprocess import CalledProcessError
from tqdm import tqdm
from tensorboardX import SummaryWriter

import sys
sys.path.append('../../')
from utils import tensorboard_log
import pathlib


def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, demo_file, tensorboard, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    save_dir = logger.get_dir()

    logger.info("Training...")
    best_success_rate = -1
    best_success_epoch = 0

    if policy.bc_loss == 1:
        print('Initializing demonstration buffer...')
        # policy.initDemoBuffer(demo_file)  #initializwe demo buffer
        policy.initDemoBuffer(demo_file, load_from_pickle=True, pickle_file='demo_pickandplace.pkl')

    for epoch in tqdm(range(n_epochs * n_cycles)):
        # train
        if epoch % n_cycles == 0:
            rollout_worker.clear_history()
        # for _ in range(n_cycles):
        logger.info("[INFO] %s" % save_dir)
        episode = rollout_worker.generate_rollouts()
        policy.store_episode(episode)

        total_q1_loss, total_pi_loss, total_cloning_loss = 0.0, 0.0, 0.0
        for _ in range(n_batches):
            q_loss, pi_loss, cloning_loss = policy.train()
            total_q1_loss += q_loss
            total_pi_loss += pi_loss
            total_cloning_loss += cloning_loss
        policy.update_target_net()

        # test
        logger.info("Testing")
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)

        log_dict = {}
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
        log_dict['cloning_loss'] = total_cloning_loss / n_batches
        if rank == 0:
            logger.dump_tabular()

        # tensorboard_log(tensorboard, log_dict, epoch)
        tensorboard_log(tensorboard, log_dict, policy.buffer.n_transitions_stored)

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            best_success_epoch = epoch
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        logger.info("Best success rate so far ", best_success_rate, " In epoch number ", best_success_epoch)
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def launch(save_policies=True):

    # Prepare params.
    params = config.DEFAULT_PARAMS
    env = params['env_name']

    logdir = os.path.join(params['root_savedir'], params['scope'], params['env_name'].split('-')[0], params['logdir'])
    if not os.path.exists(logdir):
        pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)

    # logdir = os.path.join(logdir, 'exp_{}'.format(exp_id))
    # os.mkdir(logdir)

    n_epochs = params['n_epochs']
    num_cpu = params['num_cpu']
    seed = params['seed']
    policy_save_interval = params['policy_save_interval']
    clip_return = params['clip_return']
    demo_file = params['demo_file']

    tensorboard = SummaryWriter(logdir)
    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        try:
            whoami = mpi_fork(num_cpu, ['--bind-to', 'core'])
        except CalledProcessError:
            # fancy version of mpi call failed, try simple version
            whoami = mpi_fork(num_cpu)

        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

    # params['env_name'] = env
    # params['replay_strategy'] = replay_strategy
    if env in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env])  # merge env-specific parameters in
    # params.update(**override_params)  # makes it possible to override any parameter

    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        # 'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
        #'render': 1,
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        #'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
        #'render': 1,
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorkerOriginal(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluator = RolloutWorkerOriginal(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    train(policy=policy, rollout_worker=rollout_worker,
          evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
          n_cycles=params['n_cycles'], n_batches=params['n_batches'],
          policy_save_interval=policy_save_interval, save_policies=save_policies, demo_file=demo_file,
          tensorboard=tensorboard)


def main():
    launch()


if __name__ == '__main__':
    main()
