salloc \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=128 \
  --gpus-per-task=4 \
  --constraint=gpu \
  --qos=interactive \
  --time=01:00:00 \
  --account=m4402_g \
  --image=vllm/vllm-openai:v0.8.3
