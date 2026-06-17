# scripts/diffusion_joint.py
import pickle
import random
import networkx as nx
import numpy as np
from copy import deepcopy
import os

# -----------------------------
# 1. 路径与加载
# -----------------------------
LOAD_PATH = "../datasets/hetero_graph.pkl"
SAVE_DIR = "../raw_data/diffusion_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)

with open(LOAD_PATH, "rb") as f:
    data = pickle.load(f)

G = data["graph"]
edge_type_space = data["edge_type_space"]
node_subtypes = data["node_subtypes"]
all_edge_types = data["all_edge_types"] + ["none"]

print(f"✅ 原始图载入成功：节点 {len(G.nodes())}，边 {len(G.edges())}")

# -----------------------------
# 2. 参数
# -----------------------------
num_steps = 15           # 扩散步数
noise_prob_node = 0.05   # 节点类型扰动概率
noise_prob_edge = 0.1    # 边扰动概率

nodes = list(G.nodes())

# -----------------------------
# 3. 扩散函数（节点 + 边联合）
# -----------------------------
def diffuse_joint_node_edge(G, num_steps, edge_type_space, node_subtypes,
                            noise_prob_node=0.05, noise_prob_edge=0.1):
    """
    联合扩散：节点子类型与边类型同时扩散，保持合法性。
    - 节点只在自身 node_type 的 sub_type 空间中扩散
    - 边类型只在合法的 (node_type_u, node_type_v) 空间中扩散
    """
    graphs = []
    nodes = list(G.nodes())

    for step in range(num_steps):
        G_new = deepcopy(G)

        # ---- (1) 节点子类型扩散 ----
        for n in nodes:
            if random.random() < noise_prob_node:
                ntype = G_new.nodes[n]["node_type"]
                curr_subtype = G_new.nodes[n]["sub_type"]
                candidates = [s for s in node_subtypes[ntype] if s != curr_subtype]
                if candidates:  # 确保有其他可选项
                    G_new.nodes[n]["sub_type"] = random.choice(candidates)

        # ---- (2) 边结构与类型扩散 ----
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                utype, vtype = G_new.nodes[u]["node_type"], G_new.nodes[v]["node_type"]

                # 跳过非法类型对
                if (utype, vtype) not in edge_type_space:
                    continue

                has_edge = G_new.has_edge(u, v)
                possible_types = edge_type_space[(utype, vtype)]

                # 边扰动
                if random.random() < noise_prob_edge:
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

        print(f"Step {step+1}/{num_steps} 完成: 节点数 {len(G_new.nodes())}, 边数 {len(G_new.edges())}")

    return graphs


# -----------------------------
# 4. 执行扩散并保存
# -----------------------------
print("🚀 开始节点+边联合扩散...")
diffused_graphs = diffuse_joint_node_edge(
    G,
    num_steps=num_steps,
    edge_type_space=edge_type_space,
    node_subtypes=node_subtypes,
    noise_prob_node=noise_prob_node,
    noise_prob_edge=noise_prob_edge,
)

SAVE_PATH = os.path.join(SAVE_DIR, "joint_node_edge_diffused_graphs.pkl")
with open(SAVE_PATH, "wb") as f:
    pickle.dump(diffused_graphs, f)

print(f"✅ 已保存 {len(diffused_graphs)} 个节点+边联合扩散中间图到 {SAVE_PATH}")
