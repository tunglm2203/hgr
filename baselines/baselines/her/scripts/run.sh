#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python her.py --env=FetchPush-v1 --n_epochs=200 --logdir=logs/dobot/vanilla/push/time_01 --seed 1

