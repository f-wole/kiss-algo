#!/bin/bash

## start included, end excluded ##
train_start="[1993,1,16]" # format: (year, month, day).
train_end="[2014,10,20]" # format: (year, month, day)
test_start="[2014,3,23]" # format: (year, month, day)
test_end="[2020,8,1]" # format: (year, month, day)
index="^GSPC"
path="trial"
fast="100"
slow="300"
max_loss="-0.125" # if "no" then no max_loss is applied
train_path=${path}"/data/train"
test_path=${path}"/data/test"
training_path=${path}"/training"
evaluate_train=${path}"/evaluate/train"
evaluate_test=${path}"/evaluate/test"

rm -r ${path}
mkdir -p ${train_path}
mkdir -p ${test_path}
mkdir -p ${training_path}
mkdir -p ${evaluate_train}
mkdir -p ${evaluate_test}
python get_data.py ${train_start} ${train_end} ${train_path} ${index} ${fast} ${slow}
python get_data.py ${test_start} ${test_end} ${test_path} ${index} ${fast} ${slow}
python tune.py ${train_path} ${training_path}"/params.pkl" ${max_loss}
python evaluate.py ${train_path} ${training_path}"/params.pkl" ${evaluate_train} ${fast} ${slow} ${max_loss}
python evaluate.py ${test_path} ${training_path}"/params.pkl" ${evaluate_test} ${fast} ${slow} ${max_loss}