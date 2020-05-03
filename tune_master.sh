#!/bin/bash
source /home/ec2-user/anaconda3/bin/activate tensorflow2_p36
pip install -r src/requirements.txt
tensorboard --logdir=${1}/logs --port 6012 --bind_all &
python src/tune_model.py --input_path=$1 --output_path=$2 --max_epochs=$3
