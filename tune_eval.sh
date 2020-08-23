#!/bin/bash

## start included, end excluded ##
train_start="[1980,1]" # format: (year, month).
train_end="[2002,10]" # format: (year, month)
test_start="[2016,3]" # format: (year, month)
test_end="[2020,7]" # format: (year, month)
path="trial"
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
python get_data.py ${train_start} ${train_end} ${train_path}
python get_data.py ${test_start} ${test_end} ${test_path}
python tune.py ${train_path} ${training_path}"/params.pkl"
python evaluate.py ${train_path} ${training_path}"/params.pkl" ${evaluate_train}
python evaluate.py ${test_path} ${training_path}"/params.pkl" ${evaluate_test}