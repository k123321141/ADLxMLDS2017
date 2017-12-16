#!/bin/bash
python3 ./main.py --train_dqn --dqn_dueling \
    --dqn_model=./rl/DQN_duel.h5 \
    --dqn_summary=./rl/summary/DQN_summary \
    --dqn_max_spisode=5000
    --dqn_memory=30000 
python3 ./main.py --train_dqn \
    --dqn_model=./rl/DQN_update_target100000.h5 \
    --dqn_summary=./rl/DQN_update_target100000_summary \
    --dqn_max_spisode=5000 \
    --dqn_memory=30000 
#CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.5 --dqn_double_dqn #--dqn_dueling
#CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.5 --dqn_double_dqn #--dqn_dueling
