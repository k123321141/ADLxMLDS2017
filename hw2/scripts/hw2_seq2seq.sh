#!/bin/bash
data_dir=$1
test_output_path=$2
peer_review_output_path=$3


KERAS_BACKEND=tensorflow python ./bash_parser.py $1 $2 $3
