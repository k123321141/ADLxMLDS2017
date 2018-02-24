#!/bin/bash
rm ./models/pong_ddpg*
python3 -u ./main.py --train_ddpg \
    --pg_model=./models/pong_ddpg.h5 \
    --pg_summary=./summary/pong_ddpg/summary
#CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.5 --dqn_double_dqn #--dqn_dueling
#CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.5 --dqn_double_dqn #--dqn_dueling