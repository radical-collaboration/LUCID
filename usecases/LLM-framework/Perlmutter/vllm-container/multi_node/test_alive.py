import ray

print("Before init")
ray.init(address="auto")
print("After init")

nodes = ray.nodes()
print("nodes = ", nodes)

alive = sum(n["Alive"] for n in nodes)
total = len(nodes)
print(f"Ray cluster: {alive}/{total} nodes alive.")
