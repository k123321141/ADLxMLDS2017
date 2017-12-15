#!/bin/bash
CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn \
    --dqn_model=/home/k123/rl/DQN_update_target1000.h5 \
    --dqn_summary=/home/k123/rl/DQN_update_target1000_summary \
    --dqn_max_spisode=5000

CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn \
    --dqn_model=/home/k123/rl/DQN_update_target100.h5 \
    --dqn_summary=/home/k123/rl/DQN_update_target100_summary \
    --dqn_max_spisode=5000

CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn \
    --dqn_model=/home/k123/rl/DQN_update_target50000.h5 \
    --dqn_summary=/home/k123/rl/DQN_update_target50000_summary \
    --dqn_max_spisode=5000

CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn \
    --dqn_model=/home/k123/rl/DQN_update_target1.h5 \
    --dqn_summary=/home/k123/rl/DQN_update_target1_summary \
    --dqn_max_spisode=5000
#CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.5 --dqn_double_dqn #--dqn_dueling
#CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.5 --dqn_double_dqn #--dqn_dueling
