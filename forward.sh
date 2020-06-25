#! /bin/bash

# 1 gpu id ,2 python file name, # 3 results folder name

ARRAY=(NJU2K STERE RGBD135 LFSD NLPR SIP)
# ARRAY=(m)
ELEMENTS=${#ARRAY[@]}

echo "Testing on GPU " $1  " model_dir " $2 "model_name" $3

for (( i=0;i<$ELEMENTS;i++)); do
    CUDA_VISIBLE_DEVICES=$1 python 'main.py' --mode='test' --model='./results/'$2'/'$3 --test_fold='./results/'$2'-'${3:0-6:2}'/'${ARRAY[${i}]} --sal_mode=${ARRAY[${i}]}
done

echo "Testing on e,p,d,h,s,t datasets done."
