#!/bin/bash
: '
python3 ./main.py --train_dqn \
    --dqn_model=/home/k123/rl/double_DQN.h5 \
    --dqn_summary=/home/k123/rl/double_DQN_summary \
    --dqn_double_dqn \
    --dqn_max_spisode=5000
'
python3 ./main.py --train_dqn \
    --dqn_model=./rl/duel_DQN.h5 \
    --dqn_summary=./rl/duel_DQN_summary \
    --dqn_max_spisode=5000 \
    --dqn_dueling
python3 ./main.py --train_dqn \
    --dqn_model=./rl/DQN.h5 \
    --dqn_summary=./rl/DQN_summary \
    --dqn_max_spisode=5000
#CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.5 --dqn_double_dqn #--dqn_dueling
#CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.5 --dqn_double_dqn #--dqn_dueling
