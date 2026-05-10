"""
Stage 3 — Structural Feature Computation (RQ2 / RQ3 inputs).

Computes graph-theoretic features for each OGB dataset using GPU-native
torch/PyG operations — NetworkX is avoided entirely because ogbn-products
(2.4M nodes, 62M edges) makes it infeasible.

Features computed:
  degree_mean, degree_std         — degree distribution moments
  degree_entropy                  — Shannon entropy of the degree distribution
  homophily                       — fraction of edges with matching node labels
  clustering_mean                 — mean local clustering coefficient (PyG)
  assortativity                   — Pearson r of (deg_i, deg_j) over edges
  nfd                             — Neighborhood Feature Diversity (raw features)
  shs                             — Structural Heterophily Score = 1 - homophily
  spectral_gap                    — approx. λ₂ of normalized Laplacian (power iter)

Notes:
  - ogbn-proteins has no raw input features → nfd = NaN
  - Spectral gap is approximated via 5-step power iteration; accurate enough
    for dataset ordering purposes.
  - Results are cached to --output_dir/structural_features.csv; re-running
    skips datasets that are already present unless --force is set.

Usage:
  python -m analysis.structural_features \\
      --datasets ogbn-arxiv,ogbn-mag \\
      --data_dir .data/ogb \\
      --output_dir .data/analysis \\
      [--device cuda] [--force]
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_dataset import load_dataset


# ---------------------------------------------------------------------------
# Individual feature functions (all accept Data object + device)
# ---------------------------------------------------------------------------

def compute_degree_stats(data, device):
    src = data.edge_index[0].to(device)
    deg = torch.zeros(data.num_nodes, device=device)
    deg.scatter_add_(0, src, torch.ones(src.size(0), device=device))
    return {
        "degree_mean": float(deg.mean().item()),
        "degree_std":  float(deg.std().item()),
    }


def compute_degree_entropy(data, device):
    src = data.edge_index[0].to(device)
    deg = torch.zeros(data.num_nodes, device=device)
    deg.scatter_add_(0, src, torch.ones(src.size(0), device=device))
    counts = torch.bincount(deg.long(), minlength=1)
    probs = counts.float() / counts.sum()
    probs = probs[probs > 0]
    entropy = float(-(probs * probs.log()).sum().item())
    return {"degree_entropy": entropy}


def compute_homophily(data, device):
    if data.y is None:
        return {"homophily": float("nan")}
    src = data.edge_index[0].to(device)
    dst = data.edge_index[1].to(device)
    y   = data.y.to(device)
    # Handle multi-label (ogbn-proteins): use first label column
    if y.dim() > 1:
        y = y[:, 0]
    same = (y[src] == y[dst]).float().mean().item()
    return {"homophily": float(same)}


def compute_clustering(data, device):
    """
    Mean local clustering coefficient via scipy sparse matrix multiply.

    C_v = triangles(v) / (d_v*(d_v-1)/2)
    triangles(v) = diag(A^3)[v] / 2

    We compute tri(v) = (A[v,:] * (A[v,:] @ A)).sum() for selected rows,
    which avoids materializing the full A^2 matrix. For graphs with
    >15M undirected edges we subsample 50K nodes.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from torch_geometric.utils import to_undirected, coalesce

    n  = data.num_nodes
    ei = to_undirected(data.edge_index, num_nodes=n)
    ei = coalesce(ei, num_nodes=n)

    src_np  = ei[0].cpu().numpy().astype(np.int32)
    dst_np  = ei[1].cpu().numpy().astype(np.int32)
    n_edges = ei.size(1)

    A = csr_matrix((np.ones(n_edges, dtype=np.float32), (src_np, dst_np)),
                   shape=(n, n))
    deg = np.array(A.sum(axis=1)).flatten()

    EDGE_LIMIT = 15_000_000
    if n_edges > EDGE_LIMIT:
        rng  = np.random.default_rng(42)
        vsel = rng.choice(n, size=min(50_000, n), replace=False)
    else:
        vsel = np.arange(n)

    A_rows  = A[vsel, :]          # (k, n) sparse
    A2_rows = A_rows.dot(A)       # (k, n) sparse  — rows of A^2
    tri2    = np.array(A2_rows.multiply(A_rows).sum(axis=1)).flatten()  # 2*triangles

    dv    = deg[vsel]
    valid = dv >= 2
    if not valid.any():
        return {"clustering_mean": float("nan")}

    c_v = np.clip((tri2[valid] / 2.0) / (dv[valid] * (dv[valid] - 1) / 2.0), 0.0, 1.0)
    return {"clustering_mean": float(c_v.mean())}


def compute_assortativity(data, device):
    """Degree assortativity: Pearson r of (deg_src, deg_dst) over all edges."""
    src = data.edge_index[0].to(device)
    dst = data.edge_index[1].to(device)
    deg = torch.zeros(data.num_nodes, device=device)
    deg.scatter_add_(0, src, torch.ones(src.size(0), device=device))
    d_src = deg[src].float()
    d_dst = deg[dst].float()
    # Pearson r
    stack = torch.stack([d_src, d_dst], dim=0)
    r = torch.corrcoef(stack)[0, 1].item()
    return {"assortativity": float(r)}


def compute_nfd(data, device):
    """
    Neighborhood Feature Diversity:
      NFD = E_v [ (1/|N_v|) Σ_{j∈N_v} ||h_j - μ_v||² ]
    where μ_v = mean of neighbor features of v.
    Requires raw node features data.x.
    """
    if data.x is None:
        return {"nfd": float("nan")}
    x   = data.x.to(device)           # (N, F)
    src = data.edge_index[0].to(device)
    dst = data.edge_index[1].to(device)
    E   = src.size(0)
    N   = data.num_nodes
    F   = x.size(1)

    # Degree
    deg = torch.zeros(N, device=device)
    deg.scatter_add_(0, src, torch.ones(E, device=device))

    # μ_v = scatter_mean of neighbor features (src is query, dst is neighbor)
    # edge (src→dst): src aggregates from dst
    # Chunk the scatter_add for mu to avoid (E, F) peak allocation on large graphs
    chunk = max(1, 512 * 1024)  # 512K edges ≈ 256 MB at F=128
    mu = torch.zeros(N, F, device=device)
    for s in range(0, E, chunk):
        e    = min(s + chunk, E)
        sidx = src[s:e]
        mu.scatter_add_(0, sidx.unsqueeze(1).expand(-1, F), x[dst[s:e]])
    valid_deg = deg.clamp(min=1)
    mu = mu / valid_deg.unsqueeze(1)

    # Squared distance per edge: ||x[dst] - μ[src]||², chunked to avoid (E, F) peak
    var_per_node = torch.zeros(N, device=device)
    for s in range(0, E, chunk):
        e    = min(s + chunk, E)
        diff = x[dst[s:e]] - mu[src[s:e]]  # (chunk, F)
        sq   = (diff * diff).sum(dim=1)     # (chunk,)
        var_per_node.scatter_add_(0, src[s:e], sq)
    var_per_node = var_per_node / valid_deg

    # Mean over nodes with degree >= 1; divide by F for scale-invariance
    has_neighbors = deg >= 1
    nfd = float(var_per_node[has_neighbors].mean().item()) / F
    return {"nfd": nfd}


def compute_spectral_gap(data, device, n_iter=5):
    """
    Approximate λ₂ of the normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    via power iteration on (I - L) = D^{-1/2} A D^{-1/2}, deflating the
    trivial eigenvector (constant / sqrt(N)).

    The dominant eigenvalue of (I-L) = 1; the second largest = 1 - λ₂.
    So λ₂ ≈ 1 - second_largest_eigenvalue_of_(I-L).
    """
    N   = data.num_nodes
    src = data.edge_index[0].to(device)
    dst = data.edge_index[1].to(device)

    deg = torch.zeros(N, device=device)
    deg.scatter_add_(0, src, torch.ones(src.size(0), device=device))
    deg_inv_sqrt = (deg + 1e-12).pow(-0.5)

    def matvec(v):
        """Compute D^{-1/2} A D^{-1/2} v via scatter."""
        v_scaled = deg_inv_sqrt * v                  # D^{-1/2} v
        out = torch.zeros(N, device=device)
        out.scatter_add_(0, src, v_scaled[dst])      # A (D^{-1/2} v)
        return deg_inv_sqrt * out                    # D^{-1/2} A D^{-1/2} v

    # Trivial eigenvector: proportional to sqrt(deg) (eigenvector of λ=1 for L)
    # deflate it from our iterate
    trivial = deg_inv_sqrt.clone()
    trivial = trivial / trivial.norm()

    torch.manual_seed(42)
    v = torch.randn(N, device=device)
    v = v - (v @ trivial) * trivial
    v = v / v.norm()

    lam = 0.0
    for _ in range(n_iter):
        Mv = matvec(v)
        lam = float((v * Mv).sum().item())
        # deflate trivial eigenvector again
        Mv = Mv - (Mv @ trivial) * trivial
        norm = Mv.norm().item()
        if norm < 1e-12:
            break
        v = Mv / norm

    # λ₂(L) ≈ 1 - lam  (lam is the approximated second eigenvalue of D^{-1/2}AD^{-1/2})
    spectral_gap = float(max(0.0, 1.0 - lam))
    return {"spectral_gap": spectral_gap}


# ---------------------------------------------------------------------------
# Per-dataset computation
# ---------------------------------------------------------------------------

FEATURE_FUNCS = [
    compute_degree_stats,
    compute_degree_entropy,
    compute_homophily,
    compute_clustering,
    compute_assortativity,
    compute_nfd,
    compute_spectral_gap,
]


def compute_all_features(data, device):
    features = {"num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1)}
    for fn in FEATURE_FUNCS:
        features.update(fn(data, device))
    # Derived: shs = 1 - homophily
    h = features.get("homophily", float("nan"))
    features["shs"] = float(1.0 - h) if math.isfinite(h) else float("nan")
    return features


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compute structural graph features for OGB datasets")
    p.add_argument("--datasets",   required=True,
                   help="Comma-separated OGB dataset names, e.g. 'ogbn-arxiv,ogbn-mag'")
    p.add_argument("--data_dir",   required=True, help="OGB data root directory")
    p.add_argument("--output_dir", required=True, help="Where to write structural_features.csv")
    p.add_argument("--device",     default=None,
                   help="'cpu' or 'cuda' (auto-detected if omitted)")
    p.add_argument("--force",      action="store_true",
                   help="Recompute features even if already cached in structural_features.csv")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "structural_features.csv")

    # Load existing cache
    if os.path.exists(out_path) and not args.force:
        df_existing = pd.read_csv(out_path)
        done = set(df_existing["dataset"].tolist())
    else:
        df_existing = pd.DataFrame()
        done = set()

    datasets = [d.strip() for d in args.datasets.split(",")]
    new_rows = []
    for name in datasets:
        if name in done and not args.force:
            print(f"[{name}] already cached, skipping (use --force to recompute)")
            continue
        print(f"[{name}] loading dataset...")
        data, _, _ = load_dataset(name, args.data_dir)
        print(f"[{name}] {data.num_nodes:,} nodes  {data.edge_index.size(1):,} edges  "
              f"features={'yes' if data.x is not None else 'no'}")
        print(f"[{name}] computing features on {device}...")
        feats = compute_all_features(data, device)
        row   = {"dataset": name, **feats}
        new_rows.append(row)
        for k, v in feats.items():
            print(f"    {k:30s} = {v:.6g}" if isinstance(v, float) else f"    {k:30s} = {v}")

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_out = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(
            subset="dataset", keep="last"
        )
        df_out.to_csv(out_path, index=False)
        print(f"\nStructural features → {out_path}  ({len(df_out)} datasets)")
    else:
        print("Nothing new to compute.")


if __name__ == "__main__":
    main()
