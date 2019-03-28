#!/usr/bin/env bash
# train
#python ddpg.py --env HalfCheetah-v2 --logdir ../../logs/ddpg --epochs 1000
python ddpg.py --env Hopper-v2 --logdir ../../logs/ddpg --epochs 1000

#test
#python test_policy.py --saved_model ../../logs/ddpg_HalfCheetah-v2/ddpg/ddpg_s0/ --use_tensorboard --logdir ../../logs/ddpg
