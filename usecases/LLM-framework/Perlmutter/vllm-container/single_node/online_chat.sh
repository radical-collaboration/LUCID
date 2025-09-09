curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/pscratch/sd/t/tianle/myWork/transformers/cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
    "messages": [
        {"role": "user", "content": "Who won the World Cup in 2018?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
}'
