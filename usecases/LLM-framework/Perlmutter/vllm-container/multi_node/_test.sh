#!/bin/bash

export HUGGINGFACE_HUB_CACHE=/pscratch/sd/t/tianle/myWork/transformers/cache
export TRANSFORMERS_CACHE=/pscratch/sd/t/tianle/myWork/transformers/cache
export HF_HOME=/pscratch/sd/t/tianle/myWork/transformers/cache
export HF_HUB_CACHE=/pscratch/sd/t/tianle/myWork/transformers/cache
export VLLM_CACHE_ROOT=/pscratch/sd/t/tianle/myWork/transformers/cache/vllm-cache

VLLM_IMAGE="vllm/vllm-openai:v0.9.1"
SHIFTER="shifter --image=$VLLM_IMAGE --module=gpu,nccl-plugin --env PYTHONUSERBASE=${SCRATCH}/python_user_temp/vllm_v0.8.3"

#Seems like we need this for ray to inference???
$SHIFTER python3 -m pip install --user pyarrow

#$SHIFTER env
#$SHIFTER which python3 
#$SHIFTER python3 -V 
#$SHIFTER which ray 
#$SHIFTER ray --help
$SHIFTER python3 -c "import pyarrow" 


#nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
#echo $nodes
#
#nodes_array=( $nodes )
#echo ${nodes_array[0]}
#echo ${nodes_array[1]}
#
#worker_num=$(($SLURM_JOB_NUM_NODES - 1))
#echo "<> Starting ${worker_num} ray workers..."
#
#
#ray_head_node="${nodes_array[0]}"
#for ((  i=1; i<=$worker_num; i++ )); do
#    node_i=${nodes_array[$i]}
#    echo "    - $i at $node_i"
#done
