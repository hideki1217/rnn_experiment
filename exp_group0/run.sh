#!/bin/bash

source /home/okumura/workspace/rnn_experiment/venv/bin/activate
SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd $SCRIPT_DIR/script

echo "start:" $0
echo `date`

function experiment_pre() {
    echo experiment_pre $1
    python main.py $1
    python lyapunov.py $1
    python spectoram.py $1 &
    python ED.py $1 &
    python learning_log.py $1 &
    wait
}

function experiment_epi() {
    echo experiment_epi $1
    python trajectory.py $1
}

experiment_pre exp0
experiment_pre exp1

experiment_epi exp0
experiment_epi exp1

echo "end:" $0
echo `date`