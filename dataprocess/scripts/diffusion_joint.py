# scripts/diffusion_joint.py
import pickle
import random
import networkx as nx
import numpy as np
from copy import deepcopy
import os

LOAD_PATH = "../data/hetero_graph.pkl"
SAVE_DIR = "../data/diffusion_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)

# -----------------------------
# 1. 加载数据集与元信息
# -----------------------------
with open(LOAD_PATH, "rb") as f:
    data = pickle.load(f)

G = data["graph"]
edge_type_space = data["edge_type_space"]
all_edge_types = data["all_edge_types"]

print(f"✅ 原始图载入成功：节点 {len(G.nodes())}，边 {len(G.edges())}")

# -----------------------------
# 2. 参数
# -----------------------------
num_steps = 15           # 联合扩散步数
noise_prob = 0.1         # 每步替换比例
all_edge_types.append("none")  # “无边”类型

nodes = list(G.nodes())
edges_present = set(G.edges())

# -----------------------------
# 3. 联合扩散
# -----------------------------
def diffuse_joint(G, num_steps, noise_prob, edge_type_space):
    graphs = []
    nodes = list(G.nodes())
    for step in range(num_steps):
        G_new = deepcopy(G)
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                utype, vtype = G_new.nodes[u]["node_type"], G_new.nodes[v]["node_type"]

                # 跳过非法类型对
                if (utype, vtype) not in edge_type_space:
                    continue

                has_edge = G_new.has_edge(u, v)
                possible_types = edge_type_space[(utype, vtype)]
                if random.random() < noise_prob:
                    if has_edge:
                        old_type = G_new[u][v]["edge_type"]
                        new_type = random.choice([t for t in possible_types if t != old_type] + ["none"])
                        if new_type == "none":
                            G_new.remove_edge(u, v)
                        else:
                            G_new[u][v]["edge_type"] = new_type
                    else:
                        new_type = random.choice(possible_types)
                        G_new.add_edge(u, v, edge_type=new_type)
        graphs.append(G_new)
    return graphs


# -----------------------------
# 4. 执行扩散与保存
# -----------------------------
print("🚀 开始联合扩散...")
diffused_graphs = diffuse_joint(G, num_steps, noise_prob, edge_type_space)

with open(os.path.join(SAVE_DIR, "joint_diffused_graphs.pkl"), "wb") as f:
    pickle.dump(diffused_graphs, f)

print(f"✅ 已保存 {len(diffused_graphs)} 个联合扩散中间图到 {SAVE_DIR}")
