"""
compute_dd_mag.py — compute DD_proxy(G) from trained GAT and GATv2 checkpoints
                    evaluated on the paper-cites-paper subgraph of ogbn-mag.

    DD_proxy(G) = degree-weighted mean_{i : deg(i) >= min_deg}  dd(i)
    dd(i)       = 1 - max(0, rho_bar_i)
    rho_bar_i   = sum_{(l,h) in V_i} rho_i^(l,h) / |V_i|     (uniform mean over valid (layer, head) pairs)
    rho_i^(l,h) = Spearman rho of GAT vs GATv2 attention over N(i), layer l, head h

DD_proxy = 0 : GAT and GATv2 rank every node's neighbours identically -> static attention suffices
DD_proxy = 1 : rankings are fully uncorrelated / anticorrelated -> maximum dynamism demand

Default evaluation target
-------------------------
Models are trained on ogbn-arxiv (128-dim Node2Vec features, node classification) and evaluated
on ogbn-mag. ogbn-mag is a heterogeneous knowledge graph (node types: paper, author, institution,
field_of_study; edge types: cites, writes, affiliated_with, has_topic). For DD computation we
extract the homogeneous paper-cites-paper subgraph (736,389 nodes, 5,416,271 directed edges).
Paper nodes carry 128-dim features that match the frozen checkpoint's in_channels exactly —
no zero-padding is required. DD_proxy remains valid because it is a relative GAT-vs-GATv2
comparison; both models see identical features.

Graph structure
---------------
The paper-cites-paper relation is directed (like ogbn-arxiv). Reverse edges are added after
subgraph sampling to produce an undirected graph, consistent with the ogbn-products treatment.
Author / institution / field-of-study nodes and their edges are not used.

Sample size
-----------
default sample_nodes = 200_000 (~27% of paper nodes).
The citation network is sparse (mean degree ~14.6); sampling 200k nodes yields ≈ 800k
induced undirected edges — well within T4 VRAM. Increasing to 300k yields ≈ 1.8M edges
(still safe). Use sample_nodes=0 for the full graph (requires ≥24 GB VRAM for GATv2).

Notes
-----
- Only paper nodes and paper-cites-paper edges are used; all other node/edge types are ignored.
- Both models run in separate forward passes; GAT attention is offloaded to CPU before the
  GATv2 forward to keep peak VRAM to one model's footprint instead of two.
- deg_l(i) is computed per-layer from the attention edge_index (self-loops included
  by GATConv/GATv2Conv); min_deg=3 therefore means >= 2 real neighbours.
  The returned deg tensor uses full-graph degree (no self-loops) as the formula
  specifies for the degree-weighted mean in Step 4.
"""

import argparse
import functools
import logging
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.utils import scatter, subgraph
from models import GATNodeClassifier, GATv2NodeClassifier

import matplotlib
matplotlib.use("Agg")   # headless — safe on Colab and local
import matplotlib.pyplot as plt
import numpy as np

# torch.load compatibility patch (needed for checkpoint loading)
_orig_load = torch.load
@functools.wraps(_orig_load)
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gat_ckpt",     default="Node Property Prediction\\Training (ogbn-arxiv)\\checkpoints\\best_gat_arxiv.pt")
    p.add_argument("--gatv2_ckpt",   default="Node Property Prediction\\Training (ogbn-arxiv)\\checkpoints\\best_gatv2_arxiv.pt")
    p.add_argument("--dataset",      default="ogbn-mag")
    p.add_argument("--data_dir",     default="./data")
    p.add_argument("--sample_nodes", type=int, default=200_000,
                   help="Random induced subgraph size over paper nodes; 0 = full graph (requires ≥24 GB VRAM)")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--device",       default="auto")
    p.add_argument("--min_deg",      type=int, default=3)
    p.add_argument("--out",          default="Node Property Prediction\\DD\\ogbn-mag\\dd_results\\dd_mag.pt")
    p.add_argument("--plot",         action="store_true")
    return p.parse_args()


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    a    = ckpt["args"]
    ModelCls = GATNodeClassifier if ckpt["model"] == "gat" else GATv2NodeClassifier
    model = ModelCls(
        in_channels=128, hidden_channels=a["hidden"], out_channels=40,
        num_layers=a["num_layers"], heads=a["heads"], dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    log.info(f"  {ckpt['model'].upper():6s}  val_acc={ckpt['val_acc']:.4f}  "
             f"layers={a['num_layers']}  heads={a['heads']}")
    return model, a["num_layers"]


def load_graph(dataset_name: str, data_dir: str,
               sample_nodes: int, seed: int, device: torch.device):
    """Load ogbn-mag via plain OGB API, extract paper-cites-paper subgraph, optionally subsample."""
    import gc
    from ogb.nodeproppred import NodePropPredDataset
    # NodePropPredDataset returns plain dicts — stable across all OGB/PyG versions.
    # PygNodePropPredDataset returns HeteroData in newer PyG but a flat Data in older builds,
    # causing KeyError on data['paper']; the dict API avoids this entirely.
    graph, _ = NodePropPredDataset(name=dataset_name, root=data_dir)[0]

    x_np      = graph['node_feat_dict']['paper']                       # numpy [736389, 128]
    ei_np     = graph['edge_index_dict'][('paper', 'cites', 'paper')]  # numpy [2, 5416271]
    num_nodes = int(graph['num_nodes_dict']['paper'])                   # 736389

    x          = torch.from_numpy(x_np).float()
    edge_index = torch.from_numpy(ei_np).long()

    log.info(f"  Paper nodes: {num_nodes:,}  |  Directed citation edges: {edge_index.size(1):,}")
    log.info(f"  Feature dim: {x.size(1)}  (matches checkpoint in_channels — no padding required)")

    if 0 < sample_nodes < num_nodes:
        log.info(f"  Sampling {sample_nodes:,} paper nodes  (seed={seed}) …")
        torch.manual_seed(seed)
        perm       = torch.randperm(num_nodes)[:sample_nodes]
        sub_ei, _  = subgraph(perm, edge_index, relabel_nodes=True,
                               num_nodes=num_nodes, return_edge_mask=False)
        x          = x[perm]
        num_nodes  = perm.size(0)
        edge_index = sub_ei

    del graph; gc.collect()

    # paper-cites-paper is directed — add reverse edges to make it undirected.
    row, col   = edge_index
    edge_index = torch.cat([torch.stack([row, col]), torch.stack([col, row])], dim=1)

    x          = x.to(device)
    edge_index = edge_index.to(device)

    log.info(f"  Nodes: {num_nodes:,}   Edges: {edge_index.size(1):,}  (undirected + self-loops added by model)")
    return x, edge_index, num_nodes


def _rank_within_group(values: Tensor, src: Tensor, num_nodes: int) -> Tensor:
    """0-indexed rank of each edge's value within its source-node group. Returns [E]."""
    E, device   = values.shape[0], values.device
    val_order   = values.argsort(stable=True)
    src_order   = src[val_order].argsort(stable=True)
    final_order = val_order[src_order]
    src_sorted  = src[final_order]
    deg         = scatter(torch.ones(E, device=device), src,
                          dim_size=num_nodes, reduce='sum').long()
    node_start  = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    node_start[1:] = deg.cumsum(0)
    pos  = torch.arange(E, device=device) - node_start[src_sorted]
    rank = torch.empty(E, dtype=torch.float, device=device)
    rank[final_order] = pos.float()
    return rank


def _pearson_within_group(ra: Tensor, rb: Tensor,
                           src: Tensor, num_nodes: int,
                           eps: float = 1e-8) -> Tensor:
    """Pearson correlation of (ra, rb) within each source-node group. Returns [N]."""
    mean_a = scatter(ra, src, dim_size=num_nodes, reduce='mean')
    mean_b = scatter(rb, src, dim_size=num_nodes, reduce='mean')
    ca, cb = ra - mean_a[src], rb - mean_b[src]
    cov    = scatter(ca * cb, src, dim_size=num_nodes, reduce='mean')
    va     = scatter(ca ** 2, src, dim_size=num_nodes, reduce='mean')
    vb     = scatter(cb ** 2, src, dim_size=num_nodes, reduce='mean')
    return (cov / (va.sqrt() * vb.sqrt() + eps)).clamp(-1.0, 1.0)


# Core DD computation

@torch.no_grad()
def compute_dd_proxy(gat_model, gatv2_model,
                     x: Tensor, edge_index: Tensor, num_nodes: int,
                     num_layers: int, min_deg: int = 3):
    """
    Run both models once, then compute DD_proxy(G).

    Returns
    -------
    DD       : float         graph-level scalar in [0, 1]
    dd       : Tensor [N]    per-node dynamism demand (NaN for excluded nodes)
    rho_bar  : Tensor [N]    per-node Spearman rho averaged over valid (layer, head) pairs
    deg      : Tensor [N]    full-graph degree (no self-loops)
    """
    # Run each model separately and offload attention to CPU before the next forward
    # pass — keeps peak VRAM to one model's footprint instead of two.
    _, _gat_attn_gpu = gat_model(x, edge_index)
    gat_attn = [(ei.cpu(), alpha.cpu()) for ei, alpha in _gat_attn_gpu]
    del _gat_attn_gpu
    torch.cuda.empty_cache()

    _, _gatv2_attn_gpu = gatv2_model(x, edge_index)
    gatv2_attn = [(ei.cpu(), alpha.cpu()) for ei, alpha in _gatv2_attn_gpu]
    del _gatv2_attn_gpu
    torch.cuda.empty_cache()

    # Full-graph degree from the subgraph edge_index (no GATConv self-loops) — Step 4 weight.
    full_deg = scatter(torch.ones(edge_index.size(1)), edge_index[0].cpu(),
                       dim_size=num_nodes, reduce='sum').long()   # [N]

    # Accumulate per-node rho sums across valid (layer, head) pairs — formula Step 2.
    # V_i = {(l, h) : deg_l(i) >= min_deg};  rho_bar_i = sum_{V_i} rho / |V_i|
    valid_rho_sum    = torch.zeros(num_nodes)
    valid_pair_count = torch.zeros(num_nodes)

    for l in range(num_layers):
        ei_l,  alpha_gat_l   = gat_attn[l]
        _,     alpha_gatv2_l = gatv2_attn[l]

        src_l         = ei_l[0].cpu()
        alpha_gat_l   = alpha_gat_l.cpu()
        alpha_gatv2_l = alpha_gatv2_l.cpu()
        H_l           = alpha_gat_l.shape[1]

        # Per-layer degree from attention edge_index (includes self-loops added by GATConv).
        deg_l   = scatter(torch.ones(src_l.shape[0]), src_l,
                          dim_size=num_nodes, reduce='sum').long()
        valid_l = (deg_l >= min_deg)

        rho_lh_list = []
        for h in range(H_l):
            ra     = _rank_within_group(alpha_gat_l[:, h], src_l, num_nodes)
            rb     = _rank_within_group(alpha_gatv2_l[:, h], src_l, num_nodes)
            rho_lh = _pearson_within_group(ra, rb, src_l, num_nodes)   # [N]
            rho_lh_list.append(rho_lh)
            valid_rho_sum    += torch.where(valid_l, rho_lh, torch.zeros_like(rho_lh))
            valid_pair_count += valid_l.float()

        # Per-layer stats for logging (head-averaged, valid nodes only)
        rho_l = torch.stack(rho_lh_list, dim=1).mean(dim=1)
        r     = rho_l[valid_l]
        log.info(f"  Layer {l}  H={H_l}  "
                 f"mean rho={r.mean():.4f}  median={r.median():.4f}  std={r.std():.4f}")

    # rho_bar: uniform mean over valid (l, h) pairs; denominator is |V_i|, not total heads
    has_valid = (valid_pair_count > 0)
    rho_bar   = torch.where(has_valid,
                            valid_rho_sum / valid_pair_count.clamp(min=1),
                            torch.full((num_nodes,), float('nan')))

    dd = 1.0 - rho_bar.clamp(min=0.0)
    dd[~has_valid] = float('nan')

    # Degree-weighted mean using full-graph degree (formula Step 4)
    w  = full_deg.float() * has_valid.float()
    DD = (dd.nan_to_num(0.0) * w).sum().item() / w.sum().item()

    return DD, dd, rho_bar, full_deg


def save_plots(dd: Tensor, rho_bar: Tensor, deg: Tensor,
               DD: float, dataset: str, sample_nodes: int, out_path: Path):
    valid     = ~dd.isnan()
    valid_dd  = dd[valid].numpy()
    valid_rho = rho_bar[valid].numpy()
    valid_deg = deg[valid].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"DD_proxy = {DD:.4f}  ({dataset}, {sample_nodes:,} paper nodes)", fontsize=13)

    axes[0].hist(valid_dd, bins=60, color="steelblue", edgecolor="none", alpha=0.85)
    axes[0].axvline(DD, color="crimson", linestyle="--", linewidth=1.5, label=f"DD={DD:.3f}")
    axes[0].set_xlabel("dd(i)"); axes[0].set_ylabel("count")
    axes[0].set_title("Per-node dynamism demand"); axes[0].legend()

    axes[1].hist(valid_rho, bins=60, color="darkorange", edgecolor="none", alpha=0.85)
    axes[1].axvline(valid_rho.mean(), color="navy", linestyle="--", linewidth=1.5,
                    label=f"mean={valid_rho.mean():.3f}")
    axes[1].set_xlabel(r"$\bar{\rho}_i$  (Spearman, GAT vs GATv2)")
    axes[1].set_ylabel("count"); axes[1].set_title("Per-node cross-model Spearman rho")
    axes[1].legend()

    step = max(1, len(valid_dd) // 20_000)
    axes[2].scatter(np.log1p(valid_deg[::step]), valid_dd[::step],
                    s=2, alpha=0.3, color="seagreen")
    axes[2].set_xlabel("log(1 + degree)"); axes[2].set_ylabel("dd(i)")
    axes[2].set_title("Dynamism demand vs degree")

    plt.tight_layout()
    plot_path = out_path.with_suffix(".png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    args = parse_args()

    log_path = Path("Node Property Prediction/DD/ogbn-mag/compute_dd_mag.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(log_path, mode="w"))

    if args.device == "auto":
        device = (torch.device("cuda")  if torch.cuda.is_available()  else
                  torch.device("mps")   if torch.backends.mps.is_available() else
                  torch.device("cpu"))
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        log.info(f"Device: {props.name}  ({props.total_memory / 1e9:.1f} GB VRAM)")
    else:
        log.info(f"Device: {device}")

    log.info("Loading checkpoints …")
    gat_model,   num_layers = load_model(args.gat_ckpt,   device)
    gatv2_model, _          = load_model(args.gatv2_ckpt, device)

    log.info(f"Loading {args.dataset} (paper-cites-paper subgraph) …")
    x, edge_index, num_nodes = load_graph(
        args.dataset, args.data_dir, args.sample_nodes, args.seed, device
    )

    log.info("Computing DD_proxy …")
    DD, dd, rho_bar, deg = compute_dd_proxy(
        gat_model, gatv2_model, x, edge_index, num_nodes,
        num_layers=num_layers, min_deg=args.min_deg,
    )

    valid = ~dd.isnan()
    log.info("─" * 52)
    log.info(f"  Dataset       : {args.dataset}  (seed={args.seed})")
    log.info(f"  Valid nodes   : {valid.sum().item():,}  (deg >= {args.min_deg})")
    log.info(f"  rho_bar       : mean={rho_bar[valid].mean():.4f}  "
             f"median={rho_bar[valid].median():.4f}  "
             f"std={rho_bar[valid].std():.4f}")
    log.info(f"  DD_proxy(G)   : {DD:.4f}")
    log.info(f"  Demand level  : {'low (<0.33)' if DD < 0.33 else 'moderate (0.33-0.66)' if DD < 0.66 else 'high (>0.66)'}")
    log.info("─" * 52)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "DD_proxy":     DD,
            "dd_per_node":  dd,
            "rho_bar":      rho_bar,
            "deg":          deg,
            "dataset":      args.dataset,
            "sample_nodes": args.sample_nodes,
            "seed":         args.seed,
            "min_deg":      args.min_deg,
        }, out_path)
        log.info(f"Saved to: {out_path}")

    if args.plot and args.out:
        save_plots(dd, rho_bar, deg, DD, args.dataset, args.sample_nodes, out_path)
