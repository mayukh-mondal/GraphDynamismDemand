"""
Load trained GAT and GATv2 checkpoints, run inference on ogbl-citation2, extract attention weights, and compute distance metrics between the two models' attention distributions.

Metrics
-------
Edge-level  (alpha[e, :] is an H-dim vector across heads — NOT a distribution):
  - Cosine similarity          : cos(alpha_GAT[e,:], alpha_GATv2[e,:])

Node-level  (for node i, head h: alpha[N(i), h] is a distribution over neighbors):
  - Cosine similarity          : cos(alpha_GAT[N(i),h], alpha_GATv2[N(i),h]), averaged over heads
  - Jensen-Shannon Divergence  : JSD(alpha_GAT[N(i),h] || alpha_GATv2[N(i),h]), averaged over heads
                                  natural log units → range [0, log(2)] ≈ [0, 0.693]
  - Spearman rank correlation  : ρ between GAT and GATv2 neighbor rankings, averaged over heads
                                  undefined (NaN) for nodes with deg ≤ 1

Graph-level aggregation
-----------------------
  - Edge cosine  : uniform mean over edges
  - Node metrics : degree-weighted mean (weight = deg(i))
                   high-degree nodes have more reliable estimates and greater influence on message passing, so they deserve more weight

Notes on feature dimension:
  ogbn-arxiv    : 128-dim node features (Node2Vec embeddings)
  ogbl-citation2: 128-dim node features (same)  →  no projection needed.
"""

import argparse
import logging
import functools
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.utils import scatter, subgraph
from models import GATNodeClassifier, GATv2NodeClassifier

import ogb.linkproppred.dataset_pyg as ogb_pyg

_orig = torch.load
@functools.wraps(_orig)
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig(*args, **kwargs)

torch.load = _patched_load
ogb_pyg.torch.load = _patched_load

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/compare_attention.log"),
    ],
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gat_ckpt",    type=str, required=True)
    p.add_argument("--gatv2_ckpt",  type=str, required=True)
    p.add_argument("--device",      type=str, default="auto")
    p.add_argument("--layer",       type=int, default=0)
    p.add_argument("--data_dir",    type=str, default="./data")
    p.add_argument("--out_dir",     type=str, default="./results")
    p.add_argument("--batch_edges",  type=int, default=500_000)
    p.add_argument("--sample_nodes", type=int, default=100_000,
                   help="Random induced subgraph size (0 = full graph, not recommended)")
    return p.parse_args()


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    a    = ckpt["args"]
    model_name = ckpt["model"]

    ModelCls = GATNodeClassifier if model_name == "gat" else GATv2NodeClassifier
    model = ModelCls(
        in_channels     = 128,
        hidden_channels = a["hidden"],
        out_channels    = 40,
        num_layers      = a["num_layers"],
        heads           = a["heads"],
        dropout         = 0.0,
    ).to(device)

    model.load_state_dict(ckpt["state"])
    model.eval()
    log.info(f"  Loaded {model_name.upper():6s}  (val_acc={ckpt['val_acc']:.4f}  "
             f"epoch={ckpt['epoch']})")
    return model, model_name, a["heads"], a["num_layers"]


@torch.no_grad()
def extract_attention(model, x: Tensor, edge_index: Tensor, layer_idx: int):
    """Returns (edge_index, alpha) for the requested layer.  alpha: [E, H]."""
    _, attn_list = model(x, edge_index)
    ei, alpha    = attn_list[layer_idx]
    return ei.cpu(), alpha.cpu()


# Edge-level

def per_edge_cosine(alpha_a: Tensor, alpha_b: Tensor) -> Tensor:
    """Cosine similarity across the head dimension for each edge.  Returns [E]."""
    return F.cosine_similarity(alpha_a, alpha_b, dim=1)


# Node-level helpers

def _rank_within_group(values: Tensor, src: Tensor, num_nodes: int) -> Tensor:
    """
    For each edge e, compute the 0-indexed rank of values[e] among all edges
    sharing the same source node.

    Uses the two-stable-sorts trick:
      sort by value (stable) → stable-sort by src → edges in (src, value) order.
    Returns [E] float tensor.
    """
    E      = values.shape[0]
    device = values.device

    val_order   = values.argsort(stable=True)
    src_order   = src[val_order].argsort(stable=True)
    final_order = val_order[src_order]           # edges sorted by (src, value)

    src_sorted  = src[final_order]
    deg         = scatter(torch.ones(E, device=device), src,
                          dim_size=num_nodes, reduce='sum').long()
    node_start  = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    node_start[1:] = deg.cumsum(0)

    pos  = torch.arange(E, device=device) - node_start[src_sorted]
    rank = torch.empty(E, dtype=torch.float, device=device)
    rank[final_order] = pos.float()
    return rank                                  # [E]


def _pearson_within_group(ra: Tensor, rb: Tensor,
                           src: Tensor, num_nodes: int,
                           eps: float = 1e-8) -> Tensor:
    """Pearson correlation of (ra, rb) within each source-node group.  Returns [N]."""
    mean_a = scatter(ra, src, dim_size=num_nodes, reduce='mean')
    mean_b = scatter(rb, src, dim_size=num_nodes, reduce='mean')
    ca, cb = ra - mean_a[src], rb - mean_b[src]
    cov    = scatter(ca * cb,  src, dim_size=num_nodes, reduce='mean')
    var_a  = scatter(ca ** 2,  src, dim_size=num_nodes, reduce='mean')
    var_b  = scatter(cb ** 2,  src, dim_size=num_nodes, reduce='mean')
    return (cov / (var_a.sqrt() * var_b.sqrt() + eps)).clamp(-1.0, 1.0)


# Node-level metrics

def per_node_cosine(alpha_a: Tensor, alpha_b: Tensor,
                    src: Tensor, num_nodes: int,
                    eps: float = 1e-8) -> Tensor:
    """
    For each node i and head h, compute cosine similarity between alpha_a[N(i),h]
    and alpha_b[N(i),h] (the deg(i)-dim attention distribution vectors).
    Results are averaged over heads.  Returns [N].
    """
    dot   = scatter(alpha_a * alpha_b, src, dim=0, dim_size=num_nodes, reduce='sum')   # [N, H]
    sq_a  = scatter(alpha_a ** 2,      src, dim=0, dim_size=num_nodes, reduce='sum')   # [N, H]
    sq_b  = scatter(alpha_b ** 2,      src, dim=0, dim_size=num_nodes, reduce='sum')   # [N, H]
    cos_nh = dot / (sq_a.sqrt() * sq_b.sqrt() + eps)                                   # [N, H]
    return cos_nh.mean(dim=1)                                                           # [N]


def per_node_jsd(alpha_a: Tensor, alpha_b: Tensor,
                 src: Tensor, num_nodes: int,
                 eps: float = 1e-8) -> Tensor:
    """
    Jensen-Shannon Divergence between GAT and GATv2 attention distributions for each node.
    JSD = 0.5 * KL(p||m) + 0.5 * KL(q||m)  where m = (p+q)/2.
    Per-edge contributions are scatter-summed per source node, then averaged over heads.
    Natural log units → range [0, log(2)] ≈ [0, 0.693].  Returns [N].
    """
    m      = 0.5 * (alpha_a + alpha_b)                                                  # [E, H]
    kl_pm  = torch.special.xlogy(alpha_a, alpha_a / (m + eps))                         # [E, H]
    kl_qm  = torch.special.xlogy(alpha_b, alpha_b / (m + eps))                         # [E, H]
    jsd_eh = 0.5 * (kl_pm + kl_qm)                                                     # [E, H]
    jsd_nh = scatter(jsd_eh, src, dim=0, dim_size=num_nodes, reduce='sum')              # [N, H]
    return jsd_nh.mean(dim=1)                                                           # [N]


def per_node_spearman(alpha_a: Tensor, alpha_b: Tensor,
                      src: Tensor, num_nodes: int,
                      eps: float = 1e-8) -> Tensor:
    """
    Spearman rank correlation between GAT and GATv2 neighbor rankings for each node.
    Computed per head via vectorised within-group ranking, then averaged over heads.
    Nodes with deg ≤ 1 are left as NaN (undefined ranking).  Returns [N].
    """
    H = alpha_a.shape[1]
    rho_heads = []
    for h in range(H):
        ra  = _rank_within_group(alpha_a[:, h], src, num_nodes)
        rb  = _rank_within_group(alpha_b[:, h], src, num_nodes)
        rho = _pearson_within_group(ra, rb, src, num_nodes, eps)
        rho_heads.append(rho)
    return torch.stack(rho_heads, dim=1).mean(dim=1)   # [N]


# Aggregation & reporting

def degree_weighted_mean(scores: Tensor, deg: Tensor) -> float:
    """Weighted mean of scores, ignoring NaN/Inf, with weights = degree."""
    valid = ~(scores.isnan() | scores.isinf())
    w     = deg.float() * valid.float()
    return (scores.nan_to_num(0.0) * w).sum().item() / w.sum().item()


def summarise(name: str, scores: Tensor, deg: Tensor = None, is_similarity: bool = True):
    valid = scores[~scores.isnan() & ~scores.isinf()]
    log.info(f"  {name}")
    log.info(f"    count          : {valid.numel():>12,}")
    log.info(f"    uniform mean   : {valid.mean().item():>12.6f}")
    if deg is not None:
        dw = degree_weighted_mean(scores, deg)
        log.info(f"    degree-wtd mean: {dw:>12.6f}  ← graph-level summary")
    log.info(f"    std            : {valid.std().item():>12.6f}")
    log.info(f"    median         : {valid.median().item():>12.6f}")
    log.info(f"    min            : {valid.min().item():>12.6f}")
    log.info(f"    max            : {valid.max().item():>12.6f}")
    if is_similarity:
        log.info(f"    frac > 0.9     : {(valid > 0.9).float().mean().item():>12.4f}  (highly similar)")
        log.info(f"    frac < 0.1     : {(valid < 0.1).float().mean().item():>12.4f}  (highly dissimilar)")
    else:
        log.info(f"    frac > 0.35    : {(valid > 0.35).float().mean().item():>12.4f}  (high divergence, > half of log(2))")


if __name__ == "__main__":
    args = parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info(f"Device: {device}")

    log.info("Loading checkpoints …")
    gat_model,   _, gat_heads,   gat_layers   = load_model(args.gat_ckpt,   device)
    gatv2_model, _, gatv2_heads, gatv2_layers = load_model(args.gatv2_ckpt, device)

    if gat_heads != gatv2_heads:
        log.info(f"ERROR: Head counts differ ({gat_heads} vs {gatv2_heads}).")
        raise SystemExit(1)
    if args.layer >= min(gat_layers, gatv2_layers):
        log.info(f"ERROR: --layer {args.layer} out of range.")
        raise SystemExit(1)

    log.info("Loading ogbl-citation2 …")
    dataset    = PygLinkPropPredDataset(name="ogbl-citation2", root=args.data_dir)
    data       = dataset[0]

    row, col        = data.edge_index
    data.edge_index = torch.cat([
        torch.stack([row, col]),
        torch.stack([col, row]),
    ], dim=1)

    num_nodes  = data.num_nodes

    if args.sample_nodes > 0 and args.sample_nodes < num_nodes:
        log.info(f"Sampling {args.sample_nodes:,} random nodes (induced subgraph) …")
        perm      = torch.randperm(num_nodes)[:args.sample_nodes]
        node_mask = torch.zeros(num_nodes, dtype=torch.bool)
        node_mask[perm] = True
        sub_ei, _ = subgraph(
            perm, data.edge_index, relabel_nodes=True,
            num_nodes=num_nodes, return_edge_mask=False
        )
        x          = data.x[perm].to(device)
        edge_index = sub_ei.to(device)
        num_nodes  = perm.size(0)
    else:
        x          = data.x.to(device)
        edge_index = data.edge_index.to(device)

    log.info(f"  Nodes : {num_nodes:,}")
    log.info(f"  Edges : {edge_index.size(1):,}  (undirected)")
    log.info(f"  Feats : {x.size(1)}")

    layer = args.layer
    log.info(f"Extracting attention from layer {layer} …")

    ei_gat,   alpha_gat   = extract_attention(gat_model,   x, edge_index, layer)
    ei_gatv2, alpha_gatv2 = extract_attention(gatv2_model, x, edge_index, layer)

    if ei_gat.shape != ei_gatv2.shape:
        log.info("ERROR: Edge indices differ between models.")
        raise SystemExit(1)

    log.info(f"  alpha_GAT   shape : {tuple(alpha_gat.shape)}")
    log.info(f"  alpha_GATv2 shape : {tuple(alpha_gatv2.shape)}")

    src = ei_gat[0]   # source node for each edge (shared by both models)
    deg = scatter(torch.ones(src.shape[0]), src, dim_size=num_nodes, reduce='sum').long()  # [N]

    # Edge-level (uniform mean)
    log.info("Computing per-edge cosine similarity …")
    edge_cos = per_edge_cosine(alpha_gat, alpha_gatv2)   # [E]

    # Node-level (degree-weighted mean)
    log.info("Computing per-node cosine similarity …")
    node_cos = per_node_cosine(alpha_gat, alpha_gatv2, src, num_nodes)      # [N]

    log.info("Computing per-node Jensen-Shannon divergence …")
    node_jsd = per_node_jsd(alpha_gat, alpha_gatv2, src, num_nodes)         # [N]

    log.info("Computing per-node Spearman rank correlation …")
    node_spearman = per_node_spearman(alpha_gat, alpha_gatv2, src, num_nodes)  # [N]
    node_spearman[deg <= 1] = float('nan')   # undefined for degree-0/1 nodes

    # Summary
    log.info("═" * 60)
    log.info("  ATTENTION METRICS  (GAT vs GATv2)")
    log.info(f"  Dataset : ogbl-citation2  |  Layer : {layer}")
    log.info("═" * 60)

    log.info("\n── Edge-level  (uniform mean) ──────────────────────────────")
    summarise("Cosine similarity  [across heads]", edge_cos, deg=None, is_similarity=True)

    log.info("\n── Node-level  (degree-weighted mean) ──────────────────────")
    summarise("Cosine similarity  [over neighborhood, per head avg]",
              node_cos, deg=deg.float(), is_similarity=True)
    summarise("Jensen-Shannon Div [over neighborhood, nats, per head avg]",
              node_jsd, deg=deg.float(), is_similarity=False)
    summarise("Spearman ρ         [neighbor ranking, per head avg]",
              node_spearman, deg=deg.float(), is_similarity=True)

    # Save
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / f"attention_comparison_layer{layer}.pt"
    torch.save({
        "edge_index":    ei_gat,
        "alpha_gat":     alpha_gat,
        "alpha_gatv2":   alpha_gatv2,
        "deg":           deg,
        "edge_cos":      edge_cos,
        "node_cos":      node_cos,
        "node_jsd":      node_jsd,
        "node_spearman": node_spearman,
        "layer":         layer,
        "num_nodes":     num_nodes,
    }, out_path)
    log.info(f"\nResults saved to : {out_path}")
    log.info("Tip: load with torch.load() to inspect per-node distributions, degree correlations, etc.")
