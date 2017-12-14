#!/bin/bash
CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.01 #--dqn_dueling
