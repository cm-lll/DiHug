# scripts/visualize_graph.py
import pickle
import matplotlib.pyplot as plt
import networkx as nx

LOAD_PATH = "../datasets/hetero_graph.pkl"

# -----------------------------
# 1. 加载图
# -----------------------------
with open(LOAD_PATH, "rb") as f:
    G = pickle.load(f)

print(f"加载成功: {len(G.nodes())} 个节点, {len(G.edges())} 条边")

# -----------------------------
# 2. 布局与颜色映射
# -----------------------------
pos = nx.spring_layout(G, seed=42, k=0.35)

node_color_map = {
    "Author": "#66c2a5",                # 青绿色
    "Paper": "#fc8d62",                 # 橙色
    "Institution": "#8da0cb"            # 蓝紫色
}
edge_color_map = {
    "first_author": "#e41a1c",          # 鲜艳的红色（第一作者）
    "co_author": "#377eb8",             # 深蓝色（共同作者）
    "corresponding_author": "#4daf4a",  # 鲜绿（通讯作者）
    "professor": "#984ea3",             # 紫色（教授）
    "associate_professor": "#ff7f00",   # 橙色（副教授）
    "student": "#ffc107",               # 明亮的黄色（学生）
    "collaboration": "#a65628",         # 棕色（合作关系）
    "mentorship": "#f781bf"             # 粉红色（指导关系）
}


node_colors = [node_color_map[G.nodes[n]['node_type']] for n in G.nodes()]
edge_colors = [edge_color_map[G.edges[e]['edge_type']] for e in G.edges()]

# -----------------------------
# 3. 绘图
# -----------------------------
plt.figure(figsize=(10, 8))
nx.draw(
    G, pos,
    node_color=node_colors,
    edge_color=edge_colors,
    node_size=80,
    width=1.2,
    with_labels=False
)

# 图例
for key, color in edge_color_map.items():
    plt.plot([], [], color=color, label=key, linewidth=2)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.title("Synthetic Heterogeneous Graph (Author–Paper–Institution)")
plt.show()
