#!/bin/bash
rm ./models/pong_a3c*
rm ./summary/pong_a3c/*
CUDA_VISIBLE_DEVICES="" python3 -u ./main.py --train_a3c --a3c_worker_count 22 \
    --a3c_model=./models/pong_a3c.h5 \
    --a3c_summary=./summary/pong_a3c/summary
#CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.5 --dqn_double_dqn #--dqn_dueling
#CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.5 --dqn_double_dqn #--dqn_dueling
