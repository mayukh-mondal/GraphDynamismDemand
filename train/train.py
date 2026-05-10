"""
Stage 1 — Train GAT, GATv2, or GraphSAGE on an OGB dataset.

Usage (from project root):
  python train/train.py \\
      --dataset ogbn-arxiv --task node_classification --model gat \\
      --use_default_hparams --num_runs 10 \\
      --data_dir /data/ogb --save_dir /data/ckpts --log_dir /data/logs

All three models (gat, gatv2, sage) must be trained with the same --task so that
DD comparisons are across identical objectives.
"""

import argparse
import os
import sys
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import (
    GraphSAINTRandomWalkSampler,
    LinkNeighborLoader,
    NeighborLoader,
)
from torch_geometric.utils import negative_sampling

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.hparams import (
    AVAILABLE_TASKS,
    DATASET_DEFAULTS,
    DEFAULT_TASK,
    METRIC_KEY,
    NUM_CLASSES,
    TASK_TYPE,
)
from data.load_dataset import load_dataset
from models import (
    GATLinkPredictor,
    GATNodeClassifier,
    GATv2LinkPredictor,
    GATv2NodeClassifier,
    SAGELinkPredictor,
    SAGENodeClassifier,
)
from utils.logging_utils import get_logger


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train GAT / GATv2 / GraphSAGE on OGB datasets")

    # Identity
    p.add_argument("--dataset", type=str, required=True,
                   choices=list(DATASET_DEFAULTS.keys()),
                   help="OGB dataset name")
    p.add_argument("--task", type=str, default=None,
                   help="Task to train on. For ogbn-proteins: 'all' or '0'-'111'. "
                        "Defaults to the only available task for other datasets.")
    p.add_argument("--model", type=str, required=True, choices=["gat", "gatv2", "sage"],
                   help="Model architecture")

    # Architecture
    p.add_argument("--num_layers",       type=int,   default=None)
    p.add_argument("--hidden_channels",  type=int,   default=None)
    p.add_argument("--heads",            type=int,   default=None,
                   help="Attention heads for GAT/GATv2 (ignored for SAGE)")
    p.add_argument("--dropout",          type=float, default=None)

    # Optimisation
    p.add_argument("--lr",     type=float, default=None)
    p.add_argument("--epochs", type=int,   default=None)
    p.add_argument("--num_runs", type=int, default=10,
                   help="Independent training runs (different random seeds)")

    # Sampling
    p.add_argument("--sampler", type=str, default=None,
                   choices=["neighbor", "saint", "full"])
    p.add_argument("--batch_size",   type=int, default=None)
    p.add_argument("--num_neighbors", type=str, default=None,
                   help="Comma-separated neighbors per layer, e.g. '10,10,10'")
    p.add_argument("--saint_walk_length",    type=int,  default=2)
    p.add_argument("--saint_num_steps",      type=int,  default=30)
    p.add_argument("--saint_sample_coverage",type=int,  default=50,
                   help="Node visits per node for SAINT normalization. "
                        "0 disables normalization entirely. (default: 50)")
    p.add_argument("--saint_norm_cache_dir", type=str,  default=None,
                   help="Directory to cache SAINT normalization stats. "
                        "Reused across runs on the same dataset (skips recomputation).")
    p.add_argument("--saint_num_workers",    type=int,  default=2,
                   help="DataLoader workers for SAINT subgraph sampling. "
                        "Workers run in separate processes so CPU sampling overlaps "
                        "GPU forward/backward. 0 = serial (no overlap).")
    p.add_argument("--link_num_workers",    type=int,  default=2,
                   help="DataLoader workers for LinkNeighborLoader (ogbl-citation2 mini-batch "
                        "training). Workers pre-fetch the next batch while the GPU processes "
                        "the current one. 0 = serial (no overlap). "
                        "No effect for full-batch datasets (ogbl-collab).")

    # Paths
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root directory for OGB dataset download/cache")
    p.add_argument("--save_dir", type=str, required=True,
                   help="Root directory for checkpoint files")
    p.add_argument("--log_dir",  type=str, required=True,
                   help="Root directory for log files")

    # Convenience
    p.add_argument("--use_default_hparams", action="store_true",
                   help="Fill in None hyperparameters from configs/hparams.py dataset defaults")
    p.add_argument("--device", type=str, default=None,
                   help="'cpu' or 'cuda' (auto-detected if not specified)")
    p.add_argument("--num_cpu_threads", type=int, default=8,
                   help="PyTorch intra-op CPU threads per process (OMP/MKL). "
                        "With --saint_num_workers=2 this means 2x8=16 cores for sampling. "
                        "Raise for faster CPU-only ops; lower to reduce core contention "
                        "when running many parallel jobs.")

    # Early stopping
    p.add_argument("--patience", type=int, default=50,
                   help="Epochs without improvement before stopping early. 0 = disabled.")
    p.add_argument("--es_min_delta", type=float, default=1e-4,
                   help="Minimum absolute increase in val metric that resets patience.")

    return p.parse_args()


def apply_dataset_defaults(args):
    """Fill None args from DATASET_DEFAULTS[dataset] when --use_default_hparams is set."""
    if not args.use_default_hparams:
        return
    defaults = DATASET_DEFAULTS[args.dataset]
    for key, val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)


def validate_args(args):
    if args.task is None:
        args.task = DEFAULT_TASK[args.dataset]

    valid = AVAILABLE_TASKS[args.dataset]
    if args.task not in valid:
        raise ValueError(
            f"--task '{args.task}' is not valid for dataset '{args.dataset}'. "
            f"Choose from: {valid}"
        )

    required = ["num_layers", "hidden_channels", "lr", "epochs", "sampler"]
    missing = [r for r in required if getattr(args, r, None) is None]
    if missing:
        raise ValueError(
            f"Missing required hyperparameters: {missing}. "
            f"Pass them explicitly or use --use_default_hparams."
        )

    if args.sampler in ("neighbor", "saint") and args.batch_size is None:
        raise ValueError("--batch_size is required for sampler 'neighbor' or 'saint'.")

    if args.sampler == "neighbor" and args.num_neighbors is None:
        raise ValueError("--num_neighbors is required for sampler 'neighbor'.")

    if args.heads is None:
        args.heads = 4   # sensible default; only used by GAT/GATv2

    if args.dropout is None:
        args.dropout = 0.5

    # Parse num_neighbors string → list of ints
    if isinstance(args.num_neighbors, str):
        args.num_neighbors = [int(x) for x in args.num_neighbors.split(",")]

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-derive SAINT norm cache dir from data_dir so normalization is
    # computed once per dataset and reused across all subsequent runs.
    if args.sampler == "saint" and args.saint_norm_cache_dir is None:
        args.saint_norm_cache_dir = os.path.join(args.data_dir, "saint_norm_cache")


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(args, in_channels, out_channels, task_type):
    """Instantiate the requested model for the given task type."""
    if task_type == "node_clf":
        constructors = {
            "gat":    GATNodeClassifier,
            "gatv2":  GATv2NodeClassifier,
            "sage":   SAGENodeClassifier,
        }
        return constructors[args.model](
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=out_channels,
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout,
        )
    else:  # link_pred
        constructors = {
            "gat":    GATLinkPredictor,
            "gatv2":  GATv2LinkPredictor,
            "sage":   SAGELinkPredictor,
        }
        return constructors[args.model](
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout,
        )


def get_out_channels(args):
    if args.dataset == "ogbn-proteins":
        return 112 if args.task == "all" else 1
    return NUM_CLASSES.get(args.dataset, None)  # None for link pred (not used)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def checkpoint_path(args, run: int):
    ckpt_dir = os.path.join(args.save_dir, args.dataset, args.task)
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, f"{args.model}_run{run:02d}_best.pt")


def save_checkpoint(model, optimizer, epoch, val_metric, test_metric, run, args):
    torch.save(
        {
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch":           epoch,
            "val_metric":      val_metric,
            "test_metric":     test_metric,
            "run":             run,
            "args":            vars(args),
        },
        checkpoint_path(args, run),
    )


# ---------------------------------------------------------------------------
# OGB metric extraction
# ---------------------------------------------------------------------------

def extract_metric(results: dict, dataset_name: str) -> float:
    key = METRIC_KEY[dataset_name]
    val = results[key]
    if hasattr(val, "mean"):   # mrr_list is a tensor
        return val.mean().item()
    return float(val)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_node_clf(model, data, mask, evaluator, args, device):
    """
    Evaluate node classification on the given node mask.
    Full-batch for sampler in {full, saint}; mini-batch NeighborLoader for neighbor.
    """
    model.eval()

    if args.sampler in ("full", "saint"):
        out = model(data.x.to(device), data.edge_index.to(device)).cpu()
        out_mask = out[mask]
        y_mask = data.y[mask]
    else:
        loader = NeighborLoader(
            data,
            num_neighbors=args.num_neighbors,
            batch_size=args.batch_size * 4,
            input_nodes=mask,
            shuffle=False,
            num_workers=0,
        )
        preds, labels = [], []
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)[: batch.batch_size].cpu()
            preds.append(out)
            labels.append(batch.y[: batch.batch_size].cpu())
        out_mask = torch.cat(preds, dim=0)
        y_mask = torch.cat(labels, dim=0)

    if args.dataset == "ogbn-proteins":
        # Proteins: multi-label ROC-AUC; evaluator expects raw logits/probs
        results = evaluator.eval(
            {"y_true": y_mask.float(), "y_pred": out_mask.float()}
        )
        if args.task != "all":
            # Single-task slice: evaluator still expects (N,1) shape
            results = evaluator.eval(
                {"y_true": y_mask.view(-1, 1).float(), "y_pred": out_mask.view(-1, 1).float()}
            )
    else:
        pred = out_mask.argmax(dim=-1, keepdim=True)  # (N, 1) long
        results = evaluator.eval(
            {"y_true": y_mask.view(-1, 1), "y_pred": pred}
        )

    return extract_metric(results, args.dataset)


@torch.no_grad()
def encode_all_nodes(model, data, device, args):
    """
    Compute node embeddings for the full graph (used in link prediction evaluation).
    Full-batch when sampler == 'full'; mini-batch NeighborLoader otherwise.
    Returns a CPU tensor of shape (N, hidden_channels).
    """
    model.eval()
    if args.sampler == "full":
        return model.encode(data.x.to(device), data.edge_index.to(device)).cpu()

    # Mini-batch: iterate all nodes as seeds
    loader = NeighborLoader(
        data,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size * 4,
        shuffle=False,
        num_workers=0,
    )
    z_buf = torch.empty(data.num_nodes, args.hidden_channels)
    for batch in loader:
        batch = batch.to(device)
        out = model.encode(batch.x, batch.edge_index)[: batch.batch_size].cpu()
        z_buf[batch.n_id[: batch.batch_size]] = out
    return z_buf


def batch_dot(z, src, dst, device, batch_size=65536):
    """Dot-product edge scores computed in CPU-friendly batches."""
    scores = []
    for i in range(0, len(src), batch_size):
        s = src[i : i + batch_size]
        d = dst[i : i + batch_size]
        scores.append((z[s].to(device) * z[d].to(device)).sum(-1).cpu())
    return torch.cat(scores)


@torch.no_grad()
def eval_link_pred(model, data, split_idx, evaluator, args, device, split="valid"):
    """
    Evaluate link prediction on val or test split.
    Computes all node embeddings, then scores positive/negative edges.
    """
    model.eval()
    z = encode_all_nodes(model, data, device, args)

    if args.dataset == "ogbl-collab":
        pos_edge = torch.tensor(split_idx[split]["edge"], dtype=torch.long).t()   # (2, E)
        neg_edge = torch.tensor(split_idx[split]["edge_neg"], dtype=torch.long).t()
        pos_pred = batch_dot(z, pos_edge[0], pos_edge[1], device)
        neg_pred = batch_dot(z, neg_edge[0], neg_edge[1], device)
        results = evaluator.eval({"y_pred_pos": pos_pred, "y_pred_neg": neg_pred})

    else:  # ogbl-citation2 — MRR
        src     = torch.tensor(split_idx[split]["source_node"],   dtype=torch.long)
        dst_pos = torch.tensor(split_idx[split]["target_node"],   dtype=torch.long)
        dst_neg = torch.tensor(split_idx[split]["target_node_neg"], dtype=torch.long)  # (N, 1000)
        pos_pred = batch_dot(z, src, dst_pos, device)
        # Score each of the 1000 negatives per source
        N, K = dst_neg.shape
        neg_pred_flat = batch_dot(
            z,
            src.unsqueeze(1).expand(N, K).reshape(-1),
            dst_neg.reshape(-1),
            device,
        )
        neg_pred = neg_pred_flat.view(N, K)
        results = evaluator.eval({"y_pred_pos": pos_pred, "y_pred_neg": neg_pred})

    return extract_metric(results, args.dataset)


# ---------------------------------------------------------------------------
# Node classification training loop
# ---------------------------------------------------------------------------

def _node_clf_loss(out, y, dataset_name, task):
    """Cross-entropy for single-label; BCE for ogbn-proteins."""
    if dataset_name == "ogbn-proteins":
        if task != "all":
            # Single binary subtask: y has been sliced to (N,) float
            return F.binary_cross_entropy_with_logits(out.squeeze(-1), y.float())
        return F.binary_cross_entropy_with_logits(out, y.float())
    return F.cross_entropy(out, y.long())


def _prepare_saint_data(data, train_idx, val_idx, test_idx):
    """
    Attach boolean masks so GraphSAINTRandomWalkSampler can use them.
    The sampler propagates mask attributes to each sampled subgraph.
    """
    data = data.clone()
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    return data


def train_epoch_node_clf(model, loader_or_data, train_idx, optimizer, device, args):
    """
    One training epoch for node classification.
    Returns average loss over all training nodes seen.
    """
    model.train()
    total_loss = total_nodes = 0

    if args.sampler == "full":
        data = loader_or_data
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        y = data.y[train_idx].to(device)
        out_train = out[train_idx]
        if args.dataset == "ogbn-proteins" and args.task != "all":
            task_col = int(args.task)
            out_train = out_train[:, task_col]
            y = y[:, task_col]
        loss = _node_clf_loss(out_train, y, args.dataset, args.task)
        loss.backward()
        optimizer.step()
        return loss.item()

    if args.sampler == "saint":
        for batch in loader_or_data:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            mask = batch.train_mask
            if mask.sum() == 0:
                continue
            out_m = out[mask]
            y_m = batch.y[mask]
            if args.dataset == "ogbn-proteins" and args.task != "all":
                task_col = int(args.task)
                out_m = out_m[:, task_col]
                y_m = y_m[:, task_col]
            # Weight by node_norm for unbiased SAINT loss estimate
            node_w = batch.node_norm[mask]
            elem_loss = F.binary_cross_entropy_with_logits(
                out_m, y_m.float(), reduction="none"
            ) if args.dataset == "ogbn-proteins" else F.cross_entropy(
                out_m, y_m.long(), reduction="none"
            )
            loss = (elem_loss * node_w.unsqueeze(-1) if elem_loss.dim() > 1
                    else elem_loss * node_w).sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_nodes += int(mask.sum())
        return total_loss / max(total_nodes, 1)

    # sampler == "neighbor"
    for batch in loader_or_data:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)[: batch.batch_size]
        y = batch.y[: batch.batch_size]
        if args.dataset == "ogbn-proteins" and args.task != "all":
            task_col = int(args.task)
            out = out[:, task_col]
            y = y[:, task_col]
        loss = _node_clf_loss(out, y, args.dataset, args.task)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.batch_size
        total_nodes += batch.batch_size

    return total_loss / max(total_nodes, 1)


def run_node_clf(model, optimizer, data, split_idx, evaluator, args, device, logger, run):
    train_idx = data.train_idx
    val_idx   = data.val_idx
    test_idx  = data.test_idx

    # Build data loader / sampler
    if args.sampler == "full":
        loader = data
    elif args.sampler == "saint":
        saint_data = _prepare_saint_data(data, train_idx, val_idx, test_idx)
        norm_cache = args.saint_norm_cache_dir
        if norm_cache is not None:
            os.makedirs(norm_cache, exist_ok=True)
        loader = GraphSAINTRandomWalkSampler(
            saint_data,
            batch_size=args.batch_size,
            walk_length=args.saint_walk_length,
            num_steps=args.saint_num_steps,
            sample_coverage=args.saint_sample_coverage,
            save_dir=norm_cache,
            num_workers=args.saint_num_workers,
            # Pinned memory lets the DataLoader place finished batches in
            # page-locked CPU RAM so the H2D DMA transfer can run async.
            pin_memory=(args.saint_num_workers > 0 and device.type == "cuda"),
            # Keep worker processes alive between epochs — avoids re-forking
            # overhead at the start of every epoch.
            persistent_workers=(args.saint_num_workers > 0),
        )
    else:  # neighbor
        loader = NeighborLoader(
            data,
            num_neighbors=args.num_neighbors,
            batch_size=args.batch_size,
            input_nodes=train_idx,
            shuffle=True,
            num_workers=0,
        )

    best_val = -float("inf")
    best_test = 0.0
    no_improve = 0

    for epoch in tqdm(range(1, args.epochs + 1)):
        t0 = time.time()
        loss = train_epoch_node_clf(model, loader, train_idx, optimizer, device, args)
        val_m  = eval_node_clf(model, data, val_idx,  evaluator, args, device)
        test_m = eval_node_clf(model, data, test_idx, evaluator, args, device)
        elapsed = time.time() - t0

        logger.info(
            f"Run {run:02d} | Epoch {epoch:04d} | "
            f"Loss {loss:.4f} | Val {val_m:.4f} | Test {test_m:.4f} | "
            f"Time {elapsed:.1f}s"
        )

        if val_m > best_val + args.es_min_delta:
            best_val = val_m
            best_test = test_m
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_m, test_m, run, args)
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            logger.info(
                f"Run {run:02d} | Early stopping at epoch {epoch} "
                f"({args.patience} epochs without >{args.es_min_delta:.0e} improvement)"
            )
            break

    logger.info(f"Run {run:02d} DONE | Best Val {best_val:.4f} | Test@BestVal {best_test:.4f}")
    return best_val, best_test


# ---------------------------------------------------------------------------
# Link prediction training loop
# ---------------------------------------------------------------------------

def train_epoch_link_full(model, data, train_pos, optimizer, device):
    """Full-batch link prediction training epoch (ogbl-collab)."""
    model.train()
    optimizer.zero_grad()

    edge_idx = train_pos.to(device, non_blocking=True)
    z = model.encode(data.x.to(device, non_blocking=True), edge_idx)
    neg = negative_sampling(
        edge_index=edge_idx,
        num_nodes=data.num_nodes,
        num_neg_samples=edge_idx.size(1),
    )

    pos_out = model.decode(z, edge_idx[0], edge_idx[1])
    neg_out = model.decode(z, neg[0], neg[1])

    pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return loss.item()


def train_epoch_link_mini(model, loader, optimizer, device):
    """Mini-batch link prediction training epoch (ogbl-citation2 via LinkNeighborLoader)."""
    model.train()
    total_loss = total_edges = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        z = model.encode(batch.x, batch.edge_index)
        out = model.decode(z, batch.edge_label_index[0], batch.edge_label_index[1])
        loss = F.binary_cross_entropy_with_logits(out, batch.edge_label.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.edge_label.size(0)
        total_edges += batch.edge_label.size(0)

    return total_loss / max(total_edges, 1)


def run_link_pred(model, optimizer, data, split_idx, evaluator, args, device, logger, run):
    if args.dataset == "ogbl-collab":
        train_pos = torch.tensor(split_idx["train"]["edge"], dtype=torch.long).t()  # (2, E)
        loader = None
    else:  # ogbl-citation2
        src = torch.tensor(split_idx["train"]["source_node"], dtype=torch.long)
        dst = torch.tensor(split_idx["train"]["target_node"],  dtype=torch.long)
        train_pos_edge_index = torch.stack([src, dst], dim=0)  # (2, E)
        train_pos = train_pos_edge_index

        loader = LinkNeighborLoader(
            data,
            num_neighbors=args.num_neighbors,
            edge_label_index=train_pos_edge_index,
            edge_label=torch.ones(train_pos_edge_index.size(1)),
            neg_sampling_ratio=1.0,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.link_num_workers,
            pin_memory=(args.link_num_workers > 0 and device.type == "cuda"),
            persistent_workers=(args.link_num_workers > 0),
        )

    best_val = -float("inf")
    best_test = 0.0
    no_improve = 0

    for epoch in tqdm(range(1, args.epochs + 1)):
        t0 = time.time()

        if args.sampler == "full":
            loss = train_epoch_link_full(model, data, train_pos, optimizer, device)
        else:
            loss = train_epoch_link_mini(model, loader, optimizer, device)

        val_m  = eval_link_pred(model, data, split_idx, evaluator, args, device, split="valid")
        test_m = eval_link_pred(model, data, split_idx, evaluator, args, device, split="test")
        elapsed = time.time() - t0

        logger.info(
            f"Run {run:02d} | Epoch {epoch:04d} | "
            f"Loss {loss:.4f} | Val {val_m:.4f} | Test {test_m:.4f} | "
            f"Time {elapsed:.1f}s"
        )

        if val_m > best_val + args.es_min_delta:
            best_val = val_m
            best_test = test_m
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_m, test_m, run, args)
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            logger.info(
                f"Run {run:02d} | Early stopping at epoch {epoch} "
                f"({args.patience} epochs without >{args.es_min_delta:.0e} improvement)"
            )
            break

    logger.info(f"Run {run:02d} DONE | Best Val {best_val:.4f} | Test@BestVal {best_test:.4f}")
    return best_val, best_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    apply_dataset_defaults(args)
    validate_args(args)

    # Cap CPU threads before any tensor ops. PyTorch defaults to all cores,
    # which causes thread-explosion overhead on large servers: torch_sparse
    # random walks spawn O(num_cores) OS threads per batch, starving the GPU.
    os.environ["OMP_NUM_THREADS"] = str(args.num_cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_cpu_threads)
    torch.set_num_threads(args.num_cpu_threads)
    torch.set_num_interop_threads(max(1, args.num_cpu_threads // 2))

    device = torch.device(args.device)

    log_name = f"{args.dataset}_{args.task}_{args.model}"
    logger = get_logger(args.log_dir, log_name)
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}  Task: {args.task}  Model: {args.model}")
    logger.info(str(vars(args)))

    # Load data (Stage 0)
    logger.info("Loading dataset...")
    data, split_idx, evaluator = load_dataset(args.dataset, args.data_dir)
    logger.info(
        f"Nodes: {data.num_nodes:,}  "
        f"Edges: {data.edge_index.size(1):,}  "
        f"Features: {data.x.size(1)}"
    )

    task_type    = TASK_TYPE[args.dataset]
    in_channels  = data.x.size(1)
    out_channels = get_out_channels(args)

    # Log param-count comparison hint
    if args.model in ("gat", "gatv2") and task_type == "node_clf":
        test_model = build_model(args, in_channels, out_channels or 1, task_type)
        n_params = sum(p.numel() for p in test_model.parameters())
        logger.info(f"Model parameters: {n_params:,}  "
                    f"(compare across gat/gatv2 to verify parity)")
        del test_model

    # Run experiments
    all_val, all_test = [], []

    for run in range(args.num_runs):
        set_seed(run)
        logger.info(f"{'=' * 40}  Run {run:02d}/{args.num_runs - 1}  {'=' * 40}")

        model     = build_model(args, in_channels, out_channels or args.hidden_channels, task_type)
        model     = model.to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)

        if task_type == "node_clf":
            best_val, best_test = run_node_clf(
                model, optimizer, data, split_idx, evaluator, args, device, logger, run
            )
        else:
            best_val, best_test = run_link_pred(
                model, optimizer, data, split_idx, evaluator, args, device, logger, run
            )

        all_val.append(best_val)
        all_test.append(best_test)

    # Final summary
    logger.info("=" * 60)
    logger.info(
        f"Summary ({args.num_runs} runs) | "
        f"Val:  {np.mean(all_val):.4f} ± {np.std(all_val):.4f} | "
        f"Test: {np.mean(all_test):.4f} ± {np.std(all_test):.4f}"
    )
    for run, (v, t) in enumerate(zip(all_val, all_test)):
        logger.info(f"  Run {run:02d}: Val {v:.4f}  Test {t:.4f}")


if __name__ == "__main__":
    main()
