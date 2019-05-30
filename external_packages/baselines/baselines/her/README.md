# Prioritized Hindsight Experience Replay
For details on Hindsight Experience Replay (HER), please read the [paper](https://arxiv.org/abs/1707.01495).

## How to use run code

### Getting started
To train an agent, go to folder _rl_collections/external_packages/baselines/baselines/her_, run 
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
 
 Note: In `FetchPickAndPlace`, we use the demonstration to guide policy learning like [paper](https://arxiv.org/pdf/1709.10089.pdf).
 

## Demonstrations
We use pre-recorded demonstrations to Overcome the exploration problem in HER based Reinforcement learning.
For details, please read the [paper](https://arxiv.org/pdf/1709.10089.pdf).

We use a script for the Fetch Pick and Place task from openai-baseline to generate demonstrations for the Pick and 
Place task. To generate, go to _rl_collections/external_packages/baselines/baselines/her/experiment/data_generation_, 
execute:
```bash
python pickandplace_generate_data.py
```
This outputs ```demonstration_FetchPickAndPlace.npz``` file which is our data file.

To launch training with demonstrations (more technically, with behaviour cloning loss as an auxilliary loss), go to 
_rl_collections/external_packages/baselines/baselines/her/_, edit config in _experiment/config.py_, look into  
`DEFAULT_ENV_PARAMS` region, at `FetchPickAndPlace-v1` environment, enable `bc_loss` and `q_filter` flag to True, run 
the command:
```bash
python her.py --env=FetchPickAndPlace-v1 --n_epochs=200 
--demo_file=experiment/data_generation/demonstration_FetchPickAndPlace.npz --logdir=log
```
This will train a DDPG+HER on the `FetchPickAndPlace` environment by using previously generated demonstration data.

## Configuration

To change the configuration, please see _experiment/config.py_ and _her.py_ file. _config.py_ contains the 
configuration for environment, training batch size, hyper-parameter (such as learning rate, weight, exploration 
parameter, etc.), and flag for enable/disable module (such as behavior cloning, PER, huber loss, etc.). _her.py_ 
contains the configuration for training session such as log directory, number of epochs, log interval.

The training log (also printed in the console) is stored in _logs.txt_ that lies in `logdir`.

Thank you!

