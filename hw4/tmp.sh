#!/bin/bash
rm ./gen_img/*
rm ./models/*
nohup python2 -u ./quick.py > ./nohup.out &
