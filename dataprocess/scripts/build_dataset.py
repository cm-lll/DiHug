# scripts/build_dataset.py
import networkx as nx
import random
import numpy as np
import pickle
import os

random.seed(42)
np.random.seed(42)

SAVE_PATH = "../datasets/hetero_graph.pkl"
os.makedirs("../raw_data", exist_ok=True)

# -----------------------------
# 1. 定义节点类型与子类型
# -----------------------------
num_authors = 30
num_papers = 100
num_institutions = 10

author_subtypes = ["student", "researcher", "engineer", "professor"]
institution_subtypes = ["university", "tech_company"]
paper_subtypes = ["survey", "methodology", "application", "theoretical"]

G = nx.Graph()

# 添加作者节点
for i in range(num_authors):
    subtype = random.choices(author_subtypes, weights=[0.4, 0.3, 0.2, 0.1])[0]
    G.add_node(f"A{i}", node_type="Author", sub_type=subtype)

# 添加论文节点
for i in range(num_papers):
    subtype = random.choices(paper_subtypes, weights=[0.2, 0.3, 0.4, 0.1])[0]
    G.add_node(f"P{i}", node_type="Paper", sub_type=subtype)

# 添加机构节点
for i in range(num_institutions):
    subtype = random.choices(institution_subtypes, weights=[0.7, 0.3])[0]
    G.add_node(f"I{i}", node_type="Institution", sub_type=subtype)

# -----------------------------
# 2. 定义边类型字典
# -----------------------------
edge_types = {
    ("Author", "Paper"): ["first_author", "co_author", "corresponding_author"],
    ("Author", "Institution"): ["professor", "associate_professor", "student"],
    ("Author", "Author"): ["collaboration", "mentorship"],
}

# 同时支持反向边类型定义（方便无向图使用）
edge_types.update({(b, a): tlist for (a, b), tlist in list(edge_types.items())})

# -----------------------------
# 3. 随机生成边（带概率控制）
# -----------------------------

# A–P 边（论文合作）
for a in range(num_authors):
    papers = random.sample(range(num_papers), random.randint(1, 3))
    for p in papers:
        etype = random.choices(edge_types[("Author", "Paper")], weights=[0.1, 0.7, 0.2])[0]
        G.add_edge(f"A{a}", f"P{p}", edge_type=etype)

# A–I 边（机构归属）
for a in range(num_authors):
    if random.random() < 0.9:
        inst = random.choice(range(num_institutions))
        etype = random.choices(edge_types[("Author", "Institution")], weights=[0.2, 0.3, 0.5])[0]
        G.add_edge(f"A{a}", f"I{inst}", edge_type=etype)

# A–A 边（合作关系）
for _ in range(80):
    a1, a2 = random.sample(range(num_authors), 2)
    etype = random.choices(edge_types[("Author", "Author")], weights=[0.85, 0.15])[0]
    G.add_edge(f"A{a1}", f"A{a2}", edge_type=etype)

# -----------------------------
# 4. 保存图与元信息
# -----------------------------
dataset_info = {
    "graph": G,
    "edge_type_space": edge_types,
    "all_edge_types": list({t for ts in edge_types.values() for t in ts}),
    "node_subtypes": {
        "Author": author_subtypes,
        "Paper": paper_subtypes,
        "Institution": institution_subtypes,
    },
}

with open(SAVE_PATH, "wb") as f:
    pickle.dump(dataset_info, f)

print(f"✅ 异质图与元信息已保存到 {SAVE_PATH}")
print(f"节点数: {len(G.nodes())}, 边数: {len(G.edges())}")

author_example = list(G.nodes(data=True))[:5]
print("\n示例节点:")
for n, attr in author_example:
    print(f"  {n}: type={attr['node_type']}, subtype={attr['sub_type']}")
