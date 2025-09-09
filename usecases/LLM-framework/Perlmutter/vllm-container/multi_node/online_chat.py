from openai import OpenAI

client = OpenAI(
    api_key="EMPTY", # Can not simply remote this line!
    base_url="http://localhost:8000/v1",
)


response = client.chat.completions.create(
    model="/pscratch/sd/t/tianle/myWork/transformers/cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
    messages=[
        {"role": "user", "content": "Tell me a joke about quantum physics."}
    ],
    max_tokens=60,
    temperature=0.8
)
print(response.choices[0].message.content)
