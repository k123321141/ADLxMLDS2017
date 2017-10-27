#!/bin/bash
bash_path=$0
current_dir=${0/hw1_rnn.sh/}
data_dir=$1
output_path=$2

#KERAS_BACKEND=tensorflow py


#init npz to speed up data IO
#echo ${current_path+'myinput.py'}
#python ${current_path}'myinput.py'
python ${current_dir}'bash_parser.py' ${current_dir} ${data_dir} ${output_path}
#python ${py_path} ${current_path} $1 $2

exit 0
