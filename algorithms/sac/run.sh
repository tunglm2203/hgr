#!/usr/bin/env bash
#train
#python sac.py --env HalfCheetah-v2 --logdir ../../logs/sac --epochs 1000
python sac.py --env Hopper-v2 --logdir ../../logs/sac --epochs 1000

#test
#python test_policy.py --saved_model ../../logs/sac_HalfCheetah-v2/sac/sac_s0/ --use_tensorboard --logdir ../../logs/sac

