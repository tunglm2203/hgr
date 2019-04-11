import time
import joblib
import os
import os.path as osp
import tensorflow as tf
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
from spinup.utils.mpi_tools import mpi_statistics_scalar
import numpy as np


# function to unpack observation from gym environment
def unpack_obs(obs):
    return obs['achieved_goal'], obs['desired_goal'],\
           np.concatenate((obs['observation'], obs['desired_goal'])), \
           np.concatenate((obs['observation'], obs['achieved_goal']))


def load_policy(fpath, itr='last', deterministic=False):
    """
    Usage: Load pretrained policy from `fpath`.
    Return: Environment used to train and pretrained policy
    NOTE: Only test for policy trained by spinningup
    """
    # handle which epoch to load from
    if itr == 'last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True,
               tensor_board=None, logger_kwargs=dict(), generate_demo=False,
               demo_path=None, random_policy=False, reward_type='sparse'):
    """
        Usage: Test pretrained policy in the Environment `env`. This function can be used to
               generate demonstration from pretrained policy.
        Return: Environment used to train and pretrained policy
        NOTE: Only test for policy trained by spinningup
    """

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    if 'Fetch' in env.spec.id:
        env.env.reward_type = reward_type

    infos, info_e = [], []
    if generate_demo:
        assert demo_path is not None, "Must provide demo_path to save demonstrations"
        ep_rets = []
        obs, rews, acs = [], [], []
        obs_e, rews_e, acs_e  = [], [], []

    logger = EpochLogger(**logger_kwargs)
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    _, _, o, _ = unpack_obs(o)
    act_dim = env.action_space.shape[0]

    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        if random_policy:
            a = np.random.rand(act_dim)
        else:
            a = get_action(o)
        o_next, r, d, info = env.step(a)
        _, _, o_next, _ = unpack_obs(o_next)

        info_e.append(info)
        if generate_demo:
            obs_e.append(o)
            acs_e.append(a)
            rews_e.append(r)


        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            _, _, o, _ = unpack_obs(o)
            n += 1

            if tensor_board is not None:
                mean_ret, std_ret, max_ret, min_ret  = mpi_statistics_scalar(logger.epoch_dict['EpRet'],
                                                                             with_min_and_max=True)
                mean_len, _ = mpi_statistics_scalar(logger.epoch_dict['EpLen'])
                logs = {'Return_mean': mean_ret, 'Return_std': std_ret, 'Return_max': max_ret, 'Return_min': min_ret,
                        'Episode_len': mean_len}
                tensorboard_log(tensor_board, logs, n)

            infos.append(info_e)
            info_e = []
            if generate_demo:
                ep_rets.append(sum(rews_e))
                obs.append(np.array(obs_e))
                acs.append(np.array(acs_e))
                rews.append(np.array(rews_e))

                obs_e, rews_e, acs_e = [], [], []
        else:
            o = o_next

    if generate_demo:
        if not os.path.exists(demo_path):
            os.mkdir(demo_path)

        if 'Fetch' in env.spec.id:
            np.savez_compressed(os.path.join(demo_path, 'demonstrations_' + env.spec.id),
                                ep_rets=ep_rets, obs=obs, rews=rews, acs=acs, info=infos)
        else:
            np.savez_compressed(os.path.join(demo_path, 'demonstrations_' + env.spec.id),
                                ep_rets=ep_rets, obs=obs, rews=rews, acs=acs)

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()
    if 'Fetch' in env.spec.id:
        print('Success rate: ', compute_success_rate(infos))


def tensorboard_log(tensorboard, log_dict, index):
    """ Logging information into tensorboard """
    assert isinstance(log_dict, dict), "log_dict must be `dict` type"
    for key in log_dict.keys():
        tensorboard.add_scalar(key, log_dict[key], index)


def compute_success_rate(infos):
    """
    Computing success rate from LIST of dict `infos` get from experiment
    NOTE: Only use for FetchEnv environments
    """
    n_demos = len(infos)
    success_rate = 0.0
    if n_demos == 0:
        print('[AIM-WARNING] There are no demonstrations')
        return success_rate

    def is_success(info):
        if info[-1]['is_success'] != 0.0:
            return True
        else:
            return False

    if 'is_success' in infos[0][0]:
        for i in range(n_demos):
            success_rate += float(is_success(infos[i]))
    else:
        print('[AIM-WARNING] This kind of demonstrations cannot compute success rate!')
        return success_rate

    return success_rate / n_demos


def compute_success_rate_from_list(infos):
    """
    Computing success rate from LIST of `infos` get from experiment
    NOTE: Only use for FetchEnv environments
    """
    n_demos = len(infos)
    success_rate = 0.0
    if n_demos == 0:
        print('[AIM-WARNING] There are no demonstrations')
        return success_rate

    def is_success(info):
        if info[-1] != 0.0:
            return True
        else:
            return False

    for i in range(n_demos):
        success_rate += float(is_success(infos[i]))

    return success_rate / n_demos


def my_tensorboard_logger(tag="Train_success_rate", value=None, writer=None, steps=0):
    with tf.variable_scope("environment_info", reuse=True):
        # tung: This module only support 1 environment
        if value is not None:
            summary_1 = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            writer.add_summary(summary_1, steps)


# This main for test: load_policy(), run_policy()
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=-1)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--render', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--random_policy', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy(args.fpath,
                                  args.itr if args.itr >=0 else 'last',
                                  args.deterministic)
    run_policy(env=env, get_action=get_action, max_ep_len=args.len, num_episodes=args.episodes, render=args.render)