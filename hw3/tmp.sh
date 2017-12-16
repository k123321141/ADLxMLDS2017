#!/bin/bash
python3 ./main.py --train_dqn --dqn_dueling \
    --dqn_model=./rl/DQN_duel \
    --dqn_summary=./rl/summary/DQN_duel \

#CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.5 --dqn_double_dqn #--dqn_dueling
#CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.5 --dqn_double_dqn #--dqn_dueling
