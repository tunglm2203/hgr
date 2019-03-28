#!/usr/bin/env bash
#python trpo.py --env FetchReach-v1 --logdir ../../logs/trpo_dense --epochs 1000
#python trpo.py --env FetchPickAndPlace-v1 --logdir ../../logs/trpo_dense_1 --epochs 1000 --steps 1000
#python trpo.py --env FetchPickAndPlace-v1 --logdir ../../logs/trpo_dense_3_layer_1 --epochs 1000 --steps 4000 --hid 256 --l 3 --cpu 8
#python trpo.py --env FetchSlide-v1 --logdir ../../logs/trpo_dense --epochs 1000
#python trpo.py --env FetchPush-v1 --logdir ../../logs/trpo_dense --epochs 1000 --cpu 4

python test_policy.py --episodes 10 --generate_demo --demo_path ../imitatation_learning/data/ --saved_model ../../logs.bak/trpo_dense_FetchReach-v1/trpo/trpo_s0/ --logdir ../../logs.bak/trpo_dense_test --use_tensorboard
#python test_policy.py --episodes 1000 --env FetchPickAndPlace-v1 --logdir ../../logs/trpo_dense_random --use_tensorboard --random_policy