#!/bin/bash
rm ./gen_img/*
rm ./models/*
nohup python2 -u ./acgan.py > ./nohup.out &
