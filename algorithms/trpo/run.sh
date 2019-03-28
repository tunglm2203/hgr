#!/usr/bin/env bash
#train
#python trpo.py --env HalfCheetah-v2 --logdir ../../logs/trpo --epochs 1000
python trpo.py --env Hopper-v2 --logdir ../../logs/trpo --epochs 1000

#test
#python test_policy.py --saved_model ../../logs/trpo_HalfCheetah-v2/trpo/trpo_s0/ --use_tensorboard --logdir ../../logs/trpo