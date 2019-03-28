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
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    parser.add_argument('--env', default='FetchPickAndPlace-v1', help='environment ID')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed')
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=400)

    # For training
    parser.add_argument('--expert_path', type=str, default='../her/data_fetch_random_100.npz')
    parser.add_argument('--logdir', type=str, default='../../logs/bc_400units',
                        help='Directory to save (for train) or load (for test) model')
    parser.add_argument('--BC_max_iter', type=int, default=1e5, help='Max iteration for training BC')

    # for Evaluation
    parser.add_argument('--stochastic_policy', action='store_true',
                        help='use stochastic/deterministic policy to evaluate (default: Deterministic)')
    parser.add_argument('--save_sample', action='store_true', help='save the trajectories or not')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--n_iters', type=int, default=100, help='Number of iterations for benchmark')
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
        if verbose and iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
            val_loss, _ = lossandgrad(ob_expert, ac_expert, True)
            logger.log("Training loss: {}, Validation loss: {}".format(train_loss, val_loss))
            tensorboard.add_scalar('Validation_loss', val_loss, iter_so_far)
        tensorboard.add_scalar('Training_loss', train_loss, iter_so_far)

    savedir_fname = osp.join(ckpt_dir, task_name)
    U.save_state(savedir_fname)
    return savedir_fname


def get_task_name(args):
    task_name = '{}'.format(args.env.split("-")[0])
    return task_name


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=3)
    env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)

    if not args.test_only:
        task_name = get_task_name(args)
        args.logdir = args.logdir + '_' + task_name
        dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)

        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)

        tensorboard = SummaryWriter(args.logdir)
        learn(env,
              policy_fn,
              dataset,
              max_iters=args.BC_max_iter,
              ckpt_dir=args.logdir,
              task_name=task_name,
              verbose=True, tensorboard=tensorboard)
    else:
        save_test_dir = args.logdir + '_test'
        load_model_path = os.path.join(args.logdir, args.env.split('-')[0])
        if not os.path.exists(save_test_dir):
            os.makedirs(save_test_dir)
            
        tensorboard = SummaryWriter(save_test_dir)
        runner(env=env,
               policy_func=policy_fn,
               load_model_path=load_model_path,
               timesteps_per_batch=1024,
               number_trajs=args.n_iters,
               stochastic_policy=args.stochastic_policy,
               save=args.save_sample,
               reuse=False,
               tensorboard=tensorboard,
               logdir_running=save_test_dir)


if __name__ == '__main__':
    args = argsparser()
    main(args)
