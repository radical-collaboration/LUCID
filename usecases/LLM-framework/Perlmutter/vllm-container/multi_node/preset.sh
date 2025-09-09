#!/usr/bin/env bash

export HUGGINGFACE_HUB_CACHE=/pscratch/sd/t/tianle/myWork/transformers/cache
export TRANSFORMERS_CACHE=/pscratch/sd/t/tianle/myWork/transformers/cache
export HF_HOME=/pscratch/sd/t/tianle/myWork/transformers/cache
export HF_HUB_CACHE=/pscratch/sd/t/tianle/myWork/transformers/cache
export VLLM_CACHE_ROOT=/pscratch/sd/t/tianle/myWork/transformers/cache/vllm-cache

# Define the container image and Shifter invocation
VLLM_IMAGE="vllm/vllm-openai:v0.8.3"
SHIFTER="shifter --image=$VLLM_IMAGE --module=gpu,nccl-plugin --env PYTHONUSERBASE=${SCRATCH}/python_user_temp/vllm_v0.8.3"

$SHIFTER python3 test_alive.py  

# echo ">>> Launching vLLM serve across the Ray cluster..."
# srun --nodes=1 \
#      --ntasks=1 \
#      -w "${nodes_array[0]}" \
#      --unbuffered \
#      $SHIFTER vllm serve /pscratch/.../meta-llama/... \
#              --tensor-parallel-size 8
#

