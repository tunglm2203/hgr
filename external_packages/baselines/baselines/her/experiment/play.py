# DEPRECATED, use --play flag to baselines.run instead
import numpy as np
import pickle

from baselines import logger
from baselines.common import set_global_seeds
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
import argparse


parser = argparse.ArgumentParser(description='Reinforcement Learning')
parser.add_argument('--policy_file', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_test_rollouts', type=int, default=10)
parser.add_argument('--render', action='store_true')
parser.add_argument('--env_name', type=str, default='FetchReach-v1')
# Get argument from command line
args = parser.parse_args()
print('--------- YOUR SETTING ---------')
for arg in vars(args):
    print("%15s: %s" % (str(arg), str(getattr(args, arg))))
print("")


def main(policy_file, seed, n_test_rollouts, render):
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    # env_name = policy.info['env_name']
    env_name = args.env_name

    # Prepare params.
    params = config.DEFAULT_PARAMS
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    params = config.prepare_params(params)
    config.log_params(params, logger_input=logger)

    dims = config.configure_dims(params)

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'compute_q': True,
        'rollout_batch_size': 1,
        'render': bool(render),
    }

    for name in ['time_horizon', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(seed)

    # Run evaluation.
    evaluator.clear_history()
    for _ in range(n_test_rollouts):
        evaluator.generate_rollouts()

    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()


if __name__ == '__main__':
    main(policy_file=args.policy_file,
         seed=args.seed,
         n_test_rollouts=args.n_test_rollouts,
         render=args.render)
