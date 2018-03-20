#!/bin/bash
rm ./models/pong_ac*
#rm -r ./summary/pong_ac/*
python3 -u ./main.py --train_ac \
    --ac_model=./models/pong_ac.h5 \
    --ac_summary=./summary/pong_ac_v3
#CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.5 --dqn_double_dqn #--dqn_dueling
#CUDA_VISIBLE_DEVICES="" python3 ./main.py --train_dqn --dqn_model=/tmp/rl/123.h5 --dqn_epsilon=0.5 --dqn_double_dqn #--dqn_dueling
