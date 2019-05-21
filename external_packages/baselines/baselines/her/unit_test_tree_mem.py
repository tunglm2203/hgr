from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
import numpy as np
from baselines.her.replay_buffer_test import ReplayBuffer, PrioritizedReplayBuffer


import sys

import argparse
import numpy as np
from mpi4py import MPI

from baselines import logger
from baselines.common import set_global_seeds, tf_util
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config

from subprocess import CalledProcessError
from baselines.her.util import mpi_fork
from tqdm import tqdm

import multiprocessing, logging

mpl = multiprocessing.log_to_stderr()
mpl.setLevel(logging.INFO)


def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def test(policy,  n_batches, rollout_batch_size, demo_file):

    n_iters = int(1e4)
    logger.info("Testing Replay Buffer with: {} iterations".format(n_iters))

    num_timesteps = n_iters * 50 * rollout_batch_size
    if policy.bc_loss:
        policy.init_demo_buffer(demo_file)  # initialize demo buffer if training with demonstrations
    for iters in tqdm(range(n_iters)):
        # TUNG: Fake store episode
        episode = {
            'o': np.random.uniform(0., 1., size=(rollout_batch_size, 51, 25)),
            'u': np.random.uniform(0., 1., size=(rollout_batch_size, 50, 4)),
            'g': np.random.uniform(0., 1., size=(rollout_batch_size, 50, 3)),
            'ag': np.random.uniform(0., 1., size=(rollout_batch_size, 51, 3)),
            'info_is_success': np.zeros((rollout_batch_size, 50, 1)),
        }
        policy.store_episode(episode)

        # TUNG: Fake training
        policy.fake_train(time_step=num_timesteps)

        # cpname = multiprocessing.current_process().name
        # mpl.info("\n{}-rank{} size buffer: {}".format(cpname, rank, policy.buffer.get_current_size()))
        # TUNG: Fake update priorities
        for _ in range(n_batches):
            td_errors = np.random.uniform(0., 1., size=policy.batch_size)
            if policy.bc_loss:
                new_priorities = np.abs(td_errors[:policy.batch_size - policy.demo_batch_size]) + \
                                 policy.prioritized_replay_eps
                new_priorities_for_bc = np.abs(td_errors[policy.demo_batch_size:]) + \
                                        policy.prioritized_replay_eps
                policy.buffer.update_priorities(policy.episode_idxs, new_priorities, policy.transition_idxs)
                policy.demo_buffer.update_priorities(policy.episode_idxs_for_bc, new_priorities_for_bc,
                                                     policy.transition_idxs_for_bc)
            else:
                new_priorities = np.abs(td_errors) + policy.prioritized_replay_eps
                policy.buffer.update_priorities(policy.episode_idxs, new_priorities,
                                                policy.transition_idxs)

    return policy


def launch(env, logdir, n_epochs, num_cpu, seed=None,
           replay_strategy='future', policy_save_interval=5, clip_return=True,
           demo_file=None, override_params=None, load_path=None,
           **kwargs):
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
    params = config.prepare_params(params)

    params.update(kwargs)

    dims = config.configure_dims(params)
    total_timesteps = n_epochs * params['n_cycles'] * params['time_horizon'] * params['rollout_batch_size']
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return, total_timesteps=total_timesteps)
    if load_path is not None:
        tf_util.load_variables(load_path)

    test(policy=policy,  n_batches=params['n_batches'],
         rollout_batch_size=params['rollout_batch_size'],
         demo_file=demo_file)


parser = argparse.ArgumentParser(description='Reinforcement Learning')
parser.add_argument('--env', type=str, default='FetchPickAndPlace-v1', help='The name of the Gym environment')
parser.add_argument('--logdir', type=str, default='test', help='Path to save training informatiion (model, log, etc.)')
parser.add_argument('--n_epochs', type=int, default=50, help='The number of training epochs to run')
parser.add_argument('--num_cpu', type=int, default=1, help='The number of CPU cores to use (using MPI)')
parser.add_argument('--seed', type=int, default=0, help='Seed both the environment and the training code')
parser.add_argument('--policy_save_interval', type=int, default=5, help='If 0 only the best & latest policy be pickled')
parser.add_argument('--replay_strategy', type=str, default='future', help='"future" uses HER, "none" disables HER')
parser.add_argument('--clip_return', action='store_false', help='Whether or not returns should be clipped')
parser.add_argument('--demo_file', type=str, default='experiment/data_generation/demonstration_FetchPickAndPlace.npz', help='Demo data file path')
args = parser.parse_args()


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
           load_path=None)


# TODO: If want to test, cut this function and paste before stage_batch()
def fake_train(self, batch=None, time_step=None):
    if batch is None:
        # Don't get important weight `weights` here since it is already included in `batch`
        batch, extra_info = self.sample_batch(time_step=time_step)
        if self.use_per:
            self.episode_idxs = extra_info[0]
            self.transition_idxs = extra_info[1]
            if self.bc_loss:
                self.episode_idxs_for_bc = extra_info[3]
                self.transition_idxs_for_bc = extra_info[4]


if __name__ == '__main__':
    main()
