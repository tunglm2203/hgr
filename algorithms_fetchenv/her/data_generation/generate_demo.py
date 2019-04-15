import click
import numpy as np
import pickle

from baselines import logger
from baselines.common import set_global_seeds
import sys
sys.path.append('../experiment')
import config
from rollout import my_RolloutWorkerOriginal
import os


def compute_success_rate(infos):
    if isinstance(infos, np.ndarray):
        # Format of infor: N x T x 1, where N is batch size, T is length of episode
        n_demos = infos.shape[0]
        success_rate = 0.0
        if n_demos == 0:
            print('[AIM-WARNING] There are no demonstrations')
            return -1

        def success_or_fail(info):
            if info[-1] != 0.0:
                return True
            else:
                return False

        for i in range(n_demos):
            success_rate += float(success_or_fail(infos[i]))

        return success_rate / n_demos

    elif isinstance(infos, list):
        n_demos = len(infos)
        success_rate = 0.0
        if n_demos == 0:
            print('[AIM-WARNING] There are no demonstrations')
            return -1

        def success_or_fail(info):
            if info[-1]['is_success'] != 0.0:
                return True
            else:
                return False

        if 'is_success' in infos[0][0]:
            for i in range(n_demos):
                success_rate += float(success_or_fail(infos[i]))
        else:
            print('[AIM-WARNING] This kind of demonstrations cannot compute success rate!')
            return -1
        return success_rate / n_demos
    else:
        print('[AIM-WARNING] Not support this type of `infos`')
        return


@click.command()
@click.option('--policy_file', type=str, default='../../../logs/ddpg_her_no_BC_no_Qfilter_200_epochs_10_cpus_FetchPush/policy_best.pkl')
@click.option('--seed', type=int, default=0)
@click.option('--render', type=int, default=1)
@click.option('--n_demos', type=int, default=1000)
@click.option('--save_demo', type=str, default='')
@click.option('--use_with_ddpg_her', type=int, default=1)
def main(policy_file, seed, save_demo, render, n_demos, use_with_ddpg_her):
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    env_name = policy.info['env_name']

    # List to save demonstrations
    ep_returns = []
    actions = []
    observations = []
    rewards = []
    infos = []
    dones = []

    # Prepare params.
    params = config.DEFAULT_PARAMS
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'compute_Q': True,
        'rollout_batch_size': 1,
        'render': bool(render),
    }

    for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]

    evaluator = my_RolloutWorkerOriginal(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(seed)

    # Run evaluation.
    evaluator.clear_history()
    num_demo_success = 0
    print('[AIM-NOTIFY] Generating demonstration...')
    while True:
        full_obs, episode = evaluator.generate_rollouts()
        info_tmp = episode['info_is_success']
        success_rate = compute_success_rate(info_tmp)

        if success_rate == 1.0:
            # Process observation: concatenate of 'observation' & 'goal'
            if bool(use_with_ddpg_her):
                observations.append(np.array(full_obs))
            else:
                episode['o'] = np.concatenate((episode['o'][:, :50, :], episode['g']), axis=2)
                observations.append(episode['o'].copy())

            actions.append(np.squeeze(episode['u'].copy()))
            rewards.append(np.squeeze(episode['r'].copy()))
            ep_returns.append(np.sum((np.squeeze(episode['r']))))
            dones.append(episode['d'].copy())

            # Conver info into same format in gym
            info_e = []
            for i in range(info_tmp.shape[1]):
                info_e.append({'is_success': info_tmp[0, i, 0]})
            infos.append(info_e.copy())
            num_demo_success += 1
            print('[AIM-NOTIFY] Number of demonstrations: ', num_demo_success)

        if num_demo_success == n_demos:
            break

    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()

    fileName = os.path.join(save_demo, 'demonstration_' + env_name.split('-')[0])
    np.savez_compressed(fileName, ep_rets=ep_returns, obs=observations, rews=rewards, acs=actions, info=infos)

    # Check saved demonstration
    demo = np.load(fileName + '.npz')
    success_rate = compute_success_rate(demo['info'])
    print('Success rate of demonstration: ', success_rate)


if __name__ == '__main__':
    main()
