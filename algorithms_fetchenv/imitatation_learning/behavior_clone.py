'''
The code is used to train BC imitator, or pretrained GAIL imitator
'''

import argparse
import tempfile
import os.path as osp
import gym
import logging
from tqdm import tqdm

import tensorflow as tf

from baselines.gail import mlp_policy
from baselines import bench
from baselines import logger
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines.common.mpi_adam import MpiAdam
from run_mujoco import runner
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
from gym.spaces import Box
from tensorboardX import SummaryWriter
import numpy as np
import os


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Behavior Cloning")
    parser.add_argument('--env_id', help='environment ID', default='FetchPickAndPlace-v1')
    # parser.add_argument('--env_id', help='environment ID', default='FetchReach-v1')
    # parser.add_argument('--env_id', help='environment ID', default='Hopper-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    # parser.add_argument('--expert_path', type=str, default='data/demonstrations.npz')
    parser.add_argument('--expert_path', type=str, default='../her/data_fetch_random_100.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='../../logs/bc_400units')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=400)
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e5)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--n_iters', help='Number of iterations for benchmark', type=int, default=100)
    return parser.parse_args()


def learn(env, policy_func, dataset, optim_batch_size=128, max_iters=1e4,
          adam_epsilon=1e-5, optim_stepsize=3e-4,
          ckpt_dir=None, task_name=None,
          verbose=False, tensorboard=None):

    val_per_iter = int(max_iters/10)
    ob_dim = env.observation_space.spaces['observation'].shape[0] + \
             env.observation_space.spaces['achieved_goal'].shape[0]
    ob_space = Box(low=-10, high=-10, shape=(ob_dim,))    # tung: replace to adapt w/ FetchReachEnv
    ac_space = env.action_space

    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy

    # placeholder
    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])
    stochastic = U.get_placeholder_cached(name="stochastic")

    # TODO: change to cross-entropy
    loss = tf.reduce_mean(tf.square(ac - pi.ac))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(ac, pi.ac))

    var_list = pi.get_trainable_variables()
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    lossandgrad = U.function([ob, ac, stochastic], [loss]+[U.flatgrad(loss, var_list)])

    U.initialize()
    adam.sync()
    logger.log("Pretraining with Behavior Cloning...")
    for iter_so_far in tqdm(range(int(max_iters))):
        ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
        train_loss, g = lossandgrad(ob_expert, ac_expert, True)
        adam.update(g, optim_stepsize)
        val_loss, _ = lossandgrad(ob_expert, ac_expert, True)
        if verbose and iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
            val_loss, _ = lossandgrad(ob_expert, ac_expert, True)
            logger.log("Training loss: {}, Validation loss: {}".format(train_loss, val_loss))
        tensorboard.add_scalar('Training_loss', train_loss, iter_so_far)
        tensorboard.add_scalar('Validation_loss', val_loss, iter_so_far)

    if ckpt_dir is None:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        savedir_fname = osp.join(ckpt_dir, task_name)
    U.save_state(savedir_fname)
    return savedir_fname


def get_task_name(args):
    # task_name = 'BC'
    task_name = '{}'.format(args.env_id.split("-")[0])
    # task_name += '.traj_limitation_{}'.format(args.traj_limitation)
    # task_name += ".seed_{}".format(args.seed)
    return task_name


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=3)
    env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = args.checkpoint_dir + '_' + task_name
    dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    avg_len_list, avg_ret_list = [], []

    if not args.test_only:
        tensorboard = SummaryWriter(args.checkpoint_dir)
        learn(env,
              policy_fn,
              dataset,
              max_iters=args.BC_max_iter,
              ckpt_dir=args.checkpoint_dir,
              task_name=task_name,
              verbose=True, tensorboard=tensorboard)
    else:
        args.checkpoint_dir = args.checkpoint_dir + '_test'
        tensorboard = SummaryWriter(args.checkpoint_dir)
        avg_len, avg_ret = runner(env,
                                  policy_fn,
                                  args.checkpoint_dir,
                                  timesteps_per_batch=1024,
                                  number_trajs=args.n_iters,
                                  stochastic_policy=args.stochastic_policy,
                                  save=args.save_sample,
                                  reuse=False,
                                  tensorboard=tensorboard)

        avg_len_list.append(avg_len)
        avg_ret_list.append(avg_ret)

        np.savez_compressed(osp.join(args.checkpoint_dir, 'result_bc'), avg_len=avg_len_list, avg_ret=avg_ret_list)
        print('Average length of {} iters: {}'.format(args.n_iters, np.mean(np.array(avg_len_list))))
        print('Average return of {} iters: {}'.format(args.n_iters, np.mean(np.array(avg_ret_list))))


if __name__ == '__main__':
    args = argsparser()
    main(args)
