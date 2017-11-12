#!/bin/bash
bash_path=$0
current_dir=${0/hw1_rnn.sh/}
data_dir=$1
output_path=$2

#KERAS_BACKEND=tensorflow py


KERAS_BACKEND=tensorflow python ${current_dir}'bash_parser.py' ${current_dir} ${data_dir} ${output_path} 'rnn'

exit 0
