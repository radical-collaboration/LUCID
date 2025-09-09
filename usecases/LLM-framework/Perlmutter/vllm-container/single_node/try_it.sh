#!/usr/bin/env bash

VLLM_IMAGE="vllm/vllm-openai:v0.8.3"
SHIFTER="shifter --image=$VLLM_IMAGE --module=gpu,nccl-plugin --env PYTHONUSERBASE=${SCRATCH}/vllm_v0.8.3"


#$SHIFTER python3 online_inference.py 
$SHIFTER python3 online_chat.py 
