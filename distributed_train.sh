#!/bin/bash
NUM_PROC=$1
shift
torchrun --nproc_per_node=$NUM_PROC --master_port 28800 train.py "$@"

