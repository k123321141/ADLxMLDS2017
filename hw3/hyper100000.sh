#!/bin/bash
python3 ./main.py --train_dqn \
    --dqn_model=./rl/DQN_update_target100000.h5 \
    --dqn_summary=./rl/DQN_update_target100000_summary \
    --dqn_max_spisode=5000 \
    --dqn_memory=10000 
