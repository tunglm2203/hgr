#!/usr/bin/env bash
#train
#python ppo.py --env HalfCheetah-v2 --logdir ../../logs/ppo --epochs 1000
python ppo.py --env Hopper-v2 --logdir ../../logs/ppo --epochs 1000

#test
#python test_policy.py --saved_model ../../logs/ppo_HalfCheetah-v2/ppo/ppo_s0/ --use_tensorboard --logdir ../../logs/ppo
