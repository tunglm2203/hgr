#!/usr/bin/env bash

# Train
#python behavior_clone.py --checkpoint_dir ../../logs/trpo_FetchReach_bc

# Test
python behavior_clone.py --checkpoint_dir ../../logs/trpo_FetchReach_bc --test_only --n_iters 1000