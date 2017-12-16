#!/bin/bash
CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn \
    --dqn_model=./rl/DQN_update_target1.h5 \
    --dqn_summary=./rl/DQN_update_target1_summary \
    --dqn_update_target=1 \
    --dqn_max_spisode=5000 \
    --dqn_memory=3000 
