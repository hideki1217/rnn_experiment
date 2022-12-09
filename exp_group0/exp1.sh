#!/bin/bash

source /home/okumura/workspace/rnn_experiment/venv/bin/activate
SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd $SCRIPT_DIR/script

NAME=exp1

echo "start:" $0
echo `date`

python main.py $NAME

python lyapunov.py $NAME

python spectoram.py $NAME &
python ED.py $NAME &
python learning_log.py $NAME &
wait

python trajectory.py $NAME

echo "end:" $0
echo `date`