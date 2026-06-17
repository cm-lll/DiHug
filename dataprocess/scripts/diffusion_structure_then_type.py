# scripts/diffusion_structure_then_type.py
import pickle
import random
import networkx as nx
import numpy as np
from copy import deepcopy
import os

LOAD_PATH = "../datasets/hetero_graph.pkl"
SAVE_DIR = "../raw_data/diffusion_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)

# -----------------------------
# 1. 加载原始图
# -----------------------------
with open(LOAD_PATH, "rb") as f:
    G = pickle.load(f)

print(f"✅ 原始图载入成功：节点 {len(G.nodes())}，边 {len(G.edges())}")

# -----------------------------
# 2. 扩散参数（可调整）
# -----------------------------
num_steps_structure = 10     # 结构扩散步数
num_steps_type = 10          # 类型扩散步数
add_prob = 0.02              # 每步随机加边比例
remove_prob = 0.03           # 每步随机去边比例
type_noise_prob = 0.1        # 每步随机扰动边类型概率

# 边类型全集
all_edge_types = list(set(nx.get_edge_attributes(G, "edge_type").values()))

# -----------------------------
# 3. 结构扩散（随机扰动邻接）
# -----------------------------
def diffuse_structure(G, num_steps, add_prob, remove_prob):
    graphs = []
    nodes = list(G.nodes())
    for step in range(num_steps):
        G_new = deepcopy(G)
        edges = list(G_new.edges())

        # 随机移除部分边
        to_remove = random.sample(edges, int(remove_prob * len(edges))) if edges else []
        G_new.remove_edges_from(to_remove)

        # 随机增加部分边（不同类型节点）
        for _ in range(int(add_prob * len(edges))):
            u, v = random.sample(nodes, 2)
            if not G_new.has_edge(u, v):
                # 随机赋予类型（先用一个默认）
                etype = random.choice(all_edge_types)
                G_new.add_edge(u, v, edge_type=etype)
        graphs.append(G_new)
    return graphs

# -----------------------------
# 4. 边类型扩散（在固定结构上扰动）
# -----------------------------
def diffuse_edge_types(G, num_steps, noise_prob):
    graphs = []
    for step in range(num_steps):
        G_new = deepcopy(G)
        for u, v in G_new.edges():
            if random.random() < noise_prob:
                old_type = G_new[u][v]["edge_type"]
                new_type = random.choice([t for t in all_edge_types if t != old_type])
                G_new[u][v]["edge_type"] = new_type
        graphs.append(G_new)
    return graphs

# -----------------------------
# 5. 执行扩散
# -----------------------------
print("🚀 开始结构扩散...")
structure_graphs = diffuse_structure(G, num_steps_structure, add_prob, remove_prob)
print("🚀 开始边类型扩散...")
type_graphs = diffuse_edge_types(structure_graphs[-1], num_steps_type, type_noise_prob)

# -----------------------------
# 6. 保存结果
# -----------------------------
with open(os.path.join(SAVE_DIR, "structure_diffused_graphs.pkl"), "wb") as f:
    pickle.dump(structure_graphs, f)
with open(os.path.join(SAVE_DIR, "type_diffused_graphs.pkl"), "wb") as f:
    pickle.dump(type_graphs, f)

print(f"✅ 已保存扩散结果到 {SAVE_DIR}")
print(f"共生成 {len(structure_graphs)} 个结构扩散中间图 + {len(type_graphs)} 个类型扩散中间图。")
