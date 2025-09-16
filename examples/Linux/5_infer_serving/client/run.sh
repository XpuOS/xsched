#!/bin/bash

approach=$1
EXAMPLE_DIR=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
RESULT_DIR=${EXAMPLE_DIR}/results/raw

mkdir -p ${RESULT_DIR}

if [ -z "$approach" ]; then
    echo "Usage: $0 <standalone/triton/xsched/triton-p>"
    exit 1
fi

hp_base_args="-t 60 -s 0.1 -r 40"
lp_base_args="-t 60 -s 0 -r 40"

if [ "$approach" == "standalone" ]; then
    # use high frequency to run the model
    python3 python/eval.py $lp_base_args -m bert-high -o ${RESULT_DIR}/bert-high.standalone.txt
elif [ "$approach" == "triton" ]; then
    # run norm model in triton to avoid priority config
    python3 python/eval.py $hp_base_args -m bert-norm -o ${RESULT_DIR}/bert-high.triton.txt &
    python3 python/eval.py $lp_base_args -m bert-low -o ${RESULT_DIR}/bert-low.triton.txt
elif [ "$approach" == "xsched" ]; then
    python3 python/eval.py $hp_base_args -m bert-high -o ${RESULT_DIR}/bert-high.xsched.txt &
    python3 python/eval.py $lp_base_args -m bert-low -o ${RESULT_DIR}/bert-low.xsched.txt &
elif [ "$approach" == "triton-p" ]; then
    python3 python/eval.py $hp_base_args -m bert-high -o ${RESULT_DIR}/bert-high.triton-p.txt &
    python3 python/eval.py $lp_base_args -m bert-low -o ${RESULT_DIR}/bert-low.triton-p.txt &
else
    echo "Invalid approach: $approach"
    exit 1
fi

# Wait for all background processes to finish
wait

