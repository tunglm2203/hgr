# Hindsight Goal Ranking on Replay Buffer for Sparse Reward Environment
For details, please read the [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9391700).

## How to use run code

### Getting started
To train an agent, go to folder _rl_collections/baselines/baselines/her_, run 
command (corresponding to Reach, Pick and Place, Push, and Slide task):
```bash
python her.py --env=FetchReach-v1 --n_epochs 10 --logdir test_fetchreach
```

```bash
python her.py --env=FetchPickAndPlace-v1 --n_epochs=500 --demo_file=path/to/demo_file.npz --logdir=logs
```

```bash
python her.py --env=FetchPush-v1 --n_epochs=300 --logdir=logs
```

```bash
python her.py --env=FetchSlide-v1 --n_epochs=300 --logdir=logs
```

This will train a DDPG+HER agent on the `FetchReach`, `FetchPickAndPlace`, `FetchPush`, `FetchSlide` environment. The
 argument `n_epochs` specifies number of epoch in training, if you want to convert to number of transition, please 
 look at  _./experiment/config.py_ file, e.g. look into `DEFAULT_ENV_PARAMS` region, at `FetchPickAndPlace` 
 environment, `n_cycles=20, rollout_batch_size=2`, thus number of timesteps is `n_epochs x n_cycles x 
 rollout_batch_size x time_horizon = 500 x 20 x 2 x 50 = 1e6 (timestep)`. `logdir` specifies path to log directory to
 save training infomation (saved model, log file, etc.)
 
## Configuration

To change the configuration, please see _experiment/config.py_ and _her.py_ file. _config.py_ contains the 
configuration for environment, training batch size, hyper-parameter (such as learning rate, weight, exploration 
parameter, etc.), and flag for enable/disable module (such as behavior cloning, PER, huber loss, etc.). _her.py_ 
contains the configuration for training session such as log directory, number of epochs, log interval.

The training log (also printed in the console) is stored in _logs.txt_ that lies in `logdir`.

Thank you!

