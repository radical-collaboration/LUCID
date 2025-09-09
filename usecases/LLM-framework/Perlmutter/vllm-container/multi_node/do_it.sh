#!/usr/bin/env bash

export HUGGINGFACE_HUB_CACHE=/pscratch/sd/t/tianle/myWork/transformers/cache
export TRANSFORMERS_CACHE=/pscratch/sd/t/tianle/myWork/transformers/cache
export HF_HOME=/pscratch/sd/t/tianle/myWork/transformers/cache
export HF_HUB_CACHE=/pscratch/sd/t/tianle/myWork/transformers/cache
export VLLM_CACHE_ROOT=/pscratch/sd/t/tianle/myWork/transformers/cache/vllm-cache

VLLM_IMAGE="vllm/vllm-openai:v0.8.3"
SHIFTER="shifter --image=$VLLM_IMAGE --module=gpu,nccl-plugin --env PYTHONUSERBASE=${SCRATCH}/python_user_temp/vllm_v0.8.3"

#Seems like we need this for ray to inference???
#Might need to export???
$SHIFTER python3 -m pip install --user pyarrow pandas

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=( $nodes )

ray_head_node="${nodes_array[0]}"
echo ">>> Starting Ray head on ${ray_head_node}..."
srun --nodes=1 \
     --ntasks=1 \
     -w ${ray_head_node} \
     --unbuffered \
     $SHIFTER ray start --head --block & 

sleep 30 

echo ">>> Starting Ray worker..."
worker_num=$(($SLURM_JOB_NUM_NODES - 1))
for ((  i=1; i<=$worker_num; i++ )); do
    node_i=${nodes_array[$i]}
    echo " Ray worker - $i at $node_i"
    srun --nodes=1 --ntasks=1 -w $node_i --unbuffered \
        $SHIFTER ray start --address "${ray_head_node}:6379" --block &
done

echo ">>> Verifying Ray cluster all alive..."
ray_init_timeout=300
ray_cluster_size=$SLURM_JOB_NUM_NODES
for (( i=0; i < $ray_init_timeout; i+=5 )); do
    active_nodes=`$SHIFTER python3 -c 'import ray; ray.init(); print(sum(node["Alive"] for node in ray.nodes()))'`
    if [ $active_nodes -eq $ray_cluster_size ]; then
        echo "All ray workers are active and the ray cluster is initialized successfully."
        break
    fi
    echo "Wait for all ray workers to be active. $active_nodes/$ray_cluster_size is active"
    sleep 5s;
done

$SHIFTER python3 -c "import pyarrow; import pandas; print(pyarrow.__version__); print(pandas.__version__)"
echo ">>> Launching vLLM serve across the Ray cluster..."

#This one does not work!
#$SHIFTER vllm serve /pscratch/sd/t/tianle/myWork/transformers/cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2 \
#          --tensor-parallel-size 2 --data-parallel-size 4
          
$SHIFTER vllm serve /pscratch/sd/t/tianle/myWork/transformers/cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2 \
          --tensor-parallel-size 4 --pipeline-parallel-size 2

#This one is less stable
#$SHIFTER vllm serve /pscratch/sd/t/tianle/myWork/transformers/cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2 \
#          --tensor-parallel-size 8
