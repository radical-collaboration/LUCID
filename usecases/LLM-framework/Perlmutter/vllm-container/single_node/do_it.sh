#!/usr/bin/env bash

export HUGGINGFACE_HUB_CACHE=/pscratch/sd/t/tianle/myWork/transformers/cache
export TRANSFORMERS_CACHE=/pscratch/sd/t/tianle/myWork/transformers/cache
export HF_HOME=/pscratch/sd/t/tianle/myWork/transformers/cache
export HF_HUB_CACHE=/pscratch/sd/t/tianle/myWork/transformers/cache
export VLLM_CACHE_ROOT=/pscratch/sd/t/tianle/myWork/transformers/cache/vllm-cache

VLLM_IMAGE="vllm/vllm-openai:v0.8.3"
SHIFTER="shifter --image=$VLLM_IMAGE --module=gpu,nccl-plugin --env PYTHONUSERBASE=${SCRATCH}/vllm_v0.8.3"


#$SHIFTER vllm serve meta-llama/Llama-3.3-70B-Instruct \
#          --tensor-parallel-size 4
$SHIFTER vllm serve /pscratch/sd/t/tianle/myWork/transformers/cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2 \
          --tensor-parallel-size 4

#$SHIFTER vllm serve /pscratch/sd/t/tianle/myWork/transformers/cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2 \
#          --tensor-parallel-size 4  # common use
#          --pipeline-parallel-size  # do not use this with one node 
#          --data-parallel-size      # represents the number of model replica
#          --expert-parallel-size    # use it only for MOE model

################### Testing #######################

#$SHIFTER env
#$SHIFTER which python3 
#$SHIFTER python3 -V 
#$SHIFTER which ray 
#$SHIFTER ray --help
