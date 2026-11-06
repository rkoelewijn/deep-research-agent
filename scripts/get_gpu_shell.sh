#!/bin/bash -e

# Request an interactive GPU shell on the csedu partition for debugging
NUM_GPUS="${1:-1}"
echo "Requesting interactive csedu shell with ${NUM_GPUS} GPU(s)..."
srun --partition=csedu --gres="gpu:${NUM_GPUS}" --cpus-per-task=4 --mem=24G --time=02:00:00 --pty bash

