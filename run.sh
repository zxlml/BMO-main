#!/bin/bash
# set -e
# set -x

# virtualenv -p python3 .
# source ./bin/activate

pip install -r requirements.txt
python -m ./train_test_BMO.py
