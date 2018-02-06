#!/bin/bash
rm ./models/pong_pg.h5
python3 ./main.py --train_pg \
    --pg_model=./models/pong_pg.h5 \
    --pg_summary=./summary/pong_pg_summary_keep
#CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.5 --dqn_double_dqn #--dqn_dueling
#CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.5 --dqn_double_dqn #--dqn_dueling
