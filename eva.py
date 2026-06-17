import numpy as np
import networkx as nx
import scipy.sparse as sp

try:
    from torch_two_sample import MMDStatistic
    _HAS_TTS = True
except ImportError:
    _HAS_TTS = False

import torch
from scipy.sparse.csgraph import connected_components
import powerlaw


def _to_undirected_nx_graph(adj_or_g):
    """Convert adj (scipy / np) or nx.Graph to undirected NetworkX Graph."""
    if isinstance(adj_or_g, nx.Graph):
        G = adj_or_g.copy()
    else:
        if sp.issparse(adj_or_g):
            A = adj_or_g
        else:
            A = sp.csr_matrix(adj_or_g)
        # 去掉自环，转成无向
        A = A.tocsr()
        A.setdiag(0)
        A.eliminate_zeros()
        A = ((A + A.T) > 0).astype(int)
        G = nx.from_scipy_sparse_matrix(A)
    return G


def _largest_cc_size(adj_or_g):
    """Return size of largest connected component."""
    if isinstance(adj_or_g, nx.Graph):
        G = adj_or_g
        return len(max(nx.connected_components(G), key=len)) if G.number_of_nodes() > 0 else 0
    else:
        if sp.issparse(adj_or_g):
            A = adj_or_g.tocsr()
        else:
            A = sp.csr_matrix(adj_or_g)
        n_components, labels = connected_components(A, directed=False)
        if n_components == 0:
            return 0
        _, counts = np.unique(labels, return_counts=True)
        return int(counts.max())


def _powerlaw_alpha(adj_or_g):
    """Fit power-law exponent alpha on degree sequence (ignoring <=0)."""
    G = _to_undirected_nx_graph(adj_or_g)
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    degrees = degrees[degrees > 0]
    if len(degrees) < 10:
        return float("nan")
    fit = powerlaw.Fit(degrees, verbose=False)
    return float(fit.power_law.alpha)


def _mmd_degree_fallback(deg_real, deg_gen, sigma=None):
    """
    Fallback MMD for degree distributions when torch_two_sample is not installed.
    Uses RBF kernel; returns scalar MMD (>= 0).
    """
    x = np.asarray(deg_real, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(deg_gen, dtype=np.float64).reshape(-1, 1)
    n, m = x.shape[0], y.shape[0]
    if n == 0 or m == 0:
        return float("nan")

    if sigma is None:
        from scipy.spatial.distance import pdist
        all_pts = np.vstack([x, y])
        dists = pdist(all_pts)
        sigma = float(np.median(dists)) if len(dists) > 0 else 1.0
        if sigma <= 0:
            sigma = 1.0

    gamma = 1.0 / (2.0 * sigma ** 2)

    def _rbf_mmd2(a, b):
        aa = np.exp(-gamma * (a - a.T) ** 2).mean()
        bb = np.exp(-gamma * (b - b.T) ** 2).mean()
        ab = np.exp(-gamma * (a - b.T) ** 2).mean()
        return float(aa + bb - 2 * ab)

    mmd2 = _rbf_mmd2(x, y)
    return float(np.sqrt(max(0.0, mmd2)))


def evaluate_graph_pair(real_adj_or_g, gen_adj_or_g):
    """
    Compute HGEN-style structural metrics for a pair of graphs.

    Parameters
    ----------
    real_adj_or_g : scipy.sparse, np.ndarray, or networkx.Graph
        Real graph adjacency or nx.Graph.
    gen_adj_or_g : same type
        Generated graph adjacency or nx.Graph.

    Returns
    -------
    metrics : dict
        {
          'num_nodes_real', 'num_nodes_gen',
          'num_edges_real', 'num_edges_gen',
          'avg_degree_real', 'avg_degree_gen',
          'Clustering_real', 'Clustering_gen',
          'Triangles_real', 'Triangles_gen',
          'LCCSize_real', 'LCCSize_gen',
          'EdgeOverlapRate',
          'PowerLawAlpha_real', 'PowerLawAlpha_gen',
          'DegreeMMD',
          'DegreeAssortativity_real', 'DegreeAssortativity_gen',
        }
    """
    # Normalize to NetworkX graphs for most metrics
    G_real = _to_undirected_nx_graph(real_adj_or_g)
    G_gen  = _to_undirected_nx_graph(gen_adj_or_g)

    # Basic stats
    n_real = G_real.number_of_nodes()
    n_gen  = G_gen.number_of_nodes()
    m_real = G_real.number_of_edges()
    m_gen  = G_gen.number_of_edges()

    avg_deg_real = (2 * m_real / n_real) if n_real > 0 else 0.0
    avg_deg_gen  = (2 * m_gen  / n_gen)  if n_gen  > 0 else 0.0

    # Clustering
    Clust_real = nx.average_clustering(G_real) if n_real > 0 else float("nan")
    Clust_gen  = nx.average_clustering(G_gen)  if n_gen  > 0 else float("nan")

    # Triangles
    def _triangle_count(G):
        tri = nx.triangles(G)
        return int(sum(tri.values()) // 3)

    Tri_real = _triangle_count(G_real)
    Tri_gen  = _triangle_count(G_gen)

    # LCC size
    LCC_real = _largest_cc_size(real_adj_or_g)
    LCC_gen  = _largest_cc_size(gen_adj_or_g)

    # Edge overlap (undirected, simple)
    E_real = set(G_real.edges())
    E_gen  = set(G_gen.edges())
    inter  = E_real & E_gen
    edge_overlap = len(inter) / max(len(E_real), 1)

    # Power-law alpha
    alpha_real = _powerlaw_alpha(real_adj_or_g)
    alpha_gen  = _powerlaw_alpha(gen_adj_or_g)

    # DegreeMMD
    deg_real = sorted([d for _, d in G_real.degree()], reverse=True)
    deg_gen  = sorted([d for _, d in G_gen.degree()],  reverse=True)

    if _HAS_TTS and len(deg_real) > 0 and len(deg_gen) > 0:
        mmd_stat = MMDStatistic(len(deg_real), len(deg_gen))
        x = torch.tensor(deg_real, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(deg_gen,  dtype=torch.float32).unsqueeze(-1)
        mmd_val = float(mmd_stat(x, y, alphas=[4.], ret_matrix=False))
    else:
        mmd_val = _mmd_degree_fallback(deg_real, deg_gen)

    # Degree assortativity
    assort_real = nx.degree_assortativity_coefficient(G_real) if m_real > 0 else float("nan")
    assort_gen  = nx.degree_assortativity_coefficient(G_gen)  if m_gen  > 0 else float("nan")

    return {
        "num_nodes_real": n_real,
        "num_nodes_gen":  n_gen,
        "num_edges_real": m_real,
        "num_edges_gen":  m_gen,
        "avg_degree_real": avg_deg_real,
        "avg_degree_gen":  avg_deg_gen,
        "Clustering_real": Clust_real,
        "Clustering_gen":  Clust_gen,
        "Triangles_real":  Tri_real,
        "Triangles_gen":   Tri_gen,
        "LCCSize_real":    LCC_real,
        "LCCSize_gen":     LCC_gen,
        "EdgeOverlapRate": edge_overlap,
        "PowerLawAlpha_real": alpha_real,
        "PowerLawAlpha_gen":  alpha_gen,
        "DegreeMMD":          mmd_val,
        "DegreeAssortativity_real": assort_real,
        "DegreeAssortativity_gen":  assort_gen,
    }