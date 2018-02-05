#!/bin/bash
CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_pg \
    --pg_model=./rl/pong_pg_episode.h5 \
    --pg_summary=./rl/summary/pong_pg_episode_summary 
#CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.5 --dqn_double_dqn #--dqn_dueling
#CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.5 --dqn_double_dqn #--dqn_dueling
