#!/bin/bash
CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn \
    --dqn_model=/home/k123/rl/DQN_update_target1.h5 \
    --dqn_summary=/home/k123/rl/DQN_update_target1_summary \
    --dqn_max_spisode=5000 \
    --dqn_memory=5000 

