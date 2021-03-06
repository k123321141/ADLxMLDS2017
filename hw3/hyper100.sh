#!/bin/bash
CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn \
    --dqn_model=./rl/DQN_update_target100.h5 \
    --dqn_summary=./rl/DQN_update_target100_summary \
    --dqn_update_target=100 \
    --dqn_max_spisode=5000 \
    --dqn_memory=3000 
