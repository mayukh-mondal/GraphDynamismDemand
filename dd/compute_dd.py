"""
Stage 2 — Compute Dynamism Demand (DD) from trained GAT and GATv2 checkpoints.

Decoupled from Stage 1: takes checkpoint and data directories as CLI arguments,
loads everything independently. Does NOT import from train/train.py.

Usage (from project root):
  python dd/compute_dd.py \\
      --dataset ogbn-arxiv --task node_classification \\
      --ckpt_dir /data/ckpts --data_dir /data/ogb \\
      --output_dir /data/dd_results --log_dir /data/logs \\
      --num_runs 10 --num_seeds 500 \\
      --subsample_sizes "50,100,500,1000" \\
      --k_list "1,3,5" --layer last --device cpu

Checkpoint layout expected from Stage 1:
  {ckpt_dir}/{dataset}/{task}/gat_run{run:02d}_best.pt
  {ckpt_dir}/{dataset}/{task}/gatv2_run{run:02d}_best.pt
"""

import argparse
import csv
import json
import math
import os
import sys
from tqdm import tqdm
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.hparams import AVAILABLE_TASKS, DEFAULT_TASK, TASK_TYPE
from data.load_dataset import load_dataset
from dd.definitions import (
    aev_dd_gpu,
    cq_tki_rd_dd_gpu,
    gg_dd_gpu,
)
from extract.extract_attention import extract_attention, select_layer
from models import (
    GATLinkPredictor,
    GATNodeClassifier,
    GATv2LinkPredictor,
    GATv2NodeClassifier,
)
from subsample.ego_graph_sampler import sample_ego_graphs
from utils.logging_utils import get_logger


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute DD definitions from trained GAT / GATv2 checkpoints"
    )
    p.add_argument("--dataset", type=str, required=True,
                   choices=list(AVAILABLE_TASKS.keys()),
                   help="OGB dataset name")
    p.add_argument("--task", type=str, default=None,
                   help="Task used during training (must match Stage 1 --task)")
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="Root checkpoint directory (--save_dir from Stage 1)")
    p.add_argument("--data_dir", type=str, required=True,
                   help="OGB dataset root (--data_dir from Stage 1)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory for DD result JSON and summary CSV")
    p.add_argument("--log_dir", type=str, required=True,
                   help="Directory for log files")
    p.add_argument("--num_runs", type=int, default=10,
                   help="Number of run checkpoints to process")
    p.add_argument("--num_seeds", type=int, default=500,
                   help="K seed nodes for primary DD computation")
    p.add_argument("--subsample_sizes", type=str, default="50,100,500,1000",
                   help="Comma-separated K values for convergence analysis")
    p.add_argument("--k_list", type=str, default="1,3,5",
                   help="k values for TKI-DD (comma-separated)")
    p.add_argument("--layer", type=str, default="last",
                   help="Which layer to use: 'last', 'all', or integer index")
    p.add_argument("--device", type=str, default=None,
                   help="'cpu' or 'cuda' (auto-detected if not set)")
    p.add_argument("--num_cpu_threads", type=int, default=8,
                   help="PyTorch intra-op CPU threads (OMP/MKL). k_hop_subgraph and "
                        "other torch_sparse ops are OpenMP-parallelised; without a cap "
                        "they monopolise all cores on large servers.")
    p.add_argument("--pair_chunk", type=int, default=0,
                   help="Max pair-destination entries processed per GPU batch in "
                        "CQ/TKI/RD-DD. Reduce to cut peak VRAM: 4M≈400MB overhead, "
                        "2M≈200MB. 0 = process all at once (fastest, most memory).")
    p.add_argument("--neighbor_cap", type=int, default=0,
                   help="Cap the number of source nodes considered per destination "
                        "before pair generation. Prevents quadratic blowup from hub "
                        "nodes (e.g. d=5000 → C(5000,2)=12.5M pairs; cap=50 → 1225). "
                        "0 = no cap. Recommended: 50 on 48 GB GPUs.")
    p.add_argument("--num_hops", type=int, default=2,
                   help="Ego-graph radius for subsampling. 2 gives richer shared "
                        "neighborhoods but creates very large subgraphs on dense "
                        "graphs (e.g. ogbn-mag with 2 hops + K=500 → 3.7M edges). "
                        "Use 1 for dense graphs to keep the forward pass in VRAM.")
    return p.parse_args()


def validate_args(args):
    if args.task is None:
        args.task = DEFAULT_TASK[args.dataset]
    valid = AVAILABLE_TASKS[args.dataset]
    if args.task not in valid:
        raise ValueError(
            f"--task '{args.task}' is not valid for '{args.dataset}'. "
            f"Valid choices: {valid}"
        )
    args.subsample_sizes = [int(x) for x in args.subsample_sizes.split(",")]
    args.k_list = [int(x) for x in args.k_list.split(",")]
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model reconstruction from saved checkpoint args
# ---------------------------------------------------------------------------

def rebuild_model(saved_args, in_channels, task_type):
    """
    Reconstruct a model from the args dict stored in a Stage 1 checkpoint.
    Does NOT import from train/train.py.
    """
    model_name = saved_args["model"]
    hc  = saved_args["hidden_channels"]
    nl  = saved_args["num_layers"]
    hd  = saved_args.get("heads", 4)
    dr  = saved_args.get("dropout", 0.5)

    if task_type == "node_clf":
        dataset = saved_args["dataset"]
        task    = saved_args["task"]
        # Determine out_channels
        from configs.hparams import NUM_CLASSES
        if dataset == "ogbn-proteins":
            out_channels = 112 if task == "all" else 1
        else:
            out_channels = NUM_CLASSES[dataset]

        constructors = {
            "gat":   GATNodeClassifier,
            "gatv2": GATv2NodeClassifier,
        }
        return constructors[model_name](
            in_channels=in_channels,
            hidden_channels=hc,
            out_channels=out_channels,
            num_layers=nl,
            heads=hd,
            dropout=dr,
        )
    else:  # link_pred
        constructors = {
            "gat":   GATLinkPredictor,
            "gatv2": GATv2LinkPredictor,
        }
        return constructors[model_name](
            in_channels=in_channels,
            hidden_channels=hc,
            num_layers=nl,
            heads=hd,
            dropout=dr,
        )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def ckpt_path(ckpt_dir, dataset, task, model_name, run):
    return os.path.join(ckpt_dir, dataset, task, f"{model_name}_run{run:02d}_best.pt")


def load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# DD computation for one (run, K) pair
# ---------------------------------------------------------------------------

def compute_dd_for_subgraph(
    model_v2, model_gat, data, data_x, K, run, device, k_list, layer_spec,
    pair_chunk, neighbor_cap, num_hops, logger
):
    """
    Sample a K-seed ego-subgraph, extract attention from both models,
    compute all five DD definitions.

    data_x: feature tensor, may already be on `device` (pre-loaded) or on CPU
            with pin_memory. Passed separately so the caller can pre-load once.

    Returns a flat dict of DD values.
    """
    logger.info(f"  Sampling ego-graph: K={K} seeds, {num_hops}-hop, run={run} as RNG seed")
    subset, sub_ei, _ = sample_ego_graphs(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_seeds=K,
        num_hops=num_hops,
        seed=run,
    )
    logger.info(
        f"  Subgraph: {len(subset):,} nodes, {sub_ei.size(1):,} edges"
    )

    # GPU gather if data_x is already on device; otherwise one CPU→GPU DMA
    x_sub = data_x[subset]

    # Move sub_ei to device once — reused for both model forward passes
    sub_ei_gpu = sub_ei.to(device)

    # Extract attention — returns GPU tensors (no .cpu() inside)
    attn_v2_layers  = extract_attention(model_v2,  x_sub, sub_ei_gpu, device)
    attn_gat_layers = extract_attention(model_gat, x_sub, sub_ei_gpu, device)

    selected_v2,  layer_indices = select_layer(attn_v2_layers,  layer_spec)
    selected_gat, _             = select_layer(attn_gat_layers, layer_spec)

    n_sub = subset.size(0)
    result_by_layer = {}
    for li, alpha_v2, alpha_gat in zip(layer_indices, selected_v2, selected_gat):
        # All metrics GPU-native — no CPU transfer needed
        aev            = aev_dd_gpu(sub_ei_gpu, alpha_v2, alpha_gat, n_sub)
        gg             = gg_dd_gpu(sub_ei_gpu, alpha_v2, alpha_gat, n_sub,
                                   pair_chunk=pair_chunk, neighbor_cap=neighbor_cap)
        cq, tki, rd    = cq_tki_rd_dd_gpu(sub_ei_gpu, alpha_v2, n_sub,
                                           k_list=k_list, pair_chunk=pair_chunk,
                                           neighbor_cap=neighbor_cap)

        row = {
            "rd_dd": rd,
            "cq_dd": cq,
            "gg_dd": gg,
            "aev_dd": aev,
        }
        for k, v in tki.items():
            row[f"tki_dd_{k}"] = v

        result_by_layer[li] = row
        logger.info(
            f"  Layer {li}: RD={_fmt(rd)} CQ={_fmt(cq)} GG={_fmt(gg)} "
            f"AEV={_fmt(aev)} TKI@1={_fmt(tki.get(1))} "
            f"TKI@3={_fmt(tki.get(3))} TKI@5={_fmt(tki.get(5))}"
        )

    if layer_spec == "last":
        # Flatten — only one layer
        return result_by_layer[layer_indices[0]]
    return result_by_layer


def _fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    return f"{v:.4f}"


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _summarise(run_rows, k_list):
    """Compute mean ± std across runs for every DD metric."""
    keys = ["rd_dd", "cq_dd", "gg_dd", "aev_dd"] + [f"tki_dd_{k}" for k in k_list]
    summary = {}
    for key in keys:
        vals = [r[key] for r in run_rows if key in r and not math.isnan(r[key])]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"]  = float(np.std(vals))
        else:
            summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_std"]  = float("nan")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    validate_args(args)

    os.environ["OMP_NUM_THREADS"] = str(args.num_cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_cpu_threads)
    torch.set_num_threads(args.num_cpu_threads)
    torch.set_num_interop_threads(max(1, args.num_cpu_threads // 2))

    os.makedirs(args.output_dir, exist_ok=True)
    log_name = f"{args.dataset}_{args.task}_dd"
    logger = get_logger(args.log_dir, log_name)

    logger.info("=" * 60)
    logger.info(f"Stage 2 — DD Computation")
    logger.info(f"Dataset: {args.dataset}  Task: {args.task}")
    logger.info(f"Subsample sizes: {args.subsample_sizes}")
    logger.info(f"k_list (TKI-DD): {args.k_list}")
    logger.info(f"Layer: {args.layer}  Device: {args.device}")
    logger.info(f"pair_chunk: {args.pair_chunk}  neighbor_cap: {args.neighbor_cap}  num_hops: {args.num_hops}")
    logger.info("=" * 60)

    device = torch.device(args.device)
    task_type = TASK_TYPE[args.dataset]

    # Load graph (Stage 0)
    logger.info("Loading dataset...")
    data, _, _ = load_dataset(args.dataset, args.data_dir)
    logger.info(
        f"Nodes: {data.num_nodes:,}  "
        f"Edges: {data.edge_index.size(1):,}  "
        f"Features: {data.x.size(1)}"
    )
    in_channels = data.x.size(1)

    # Pre-load feature matrix to GPU to eliminate repeated CPU→GPU transfers
    if device.type == "cuda":
        try:
            data_x = data.x.to(device)
            logger.info(
                f"Feature matrix on GPU: {tuple(data_x.shape)}  "
                f"({data_x.numel() * 4 / 1e6:.1f} MB)"
            )
        except RuntimeError:
            data_x = data.x.pin_memory()
            logger.warning(
                "Feature matrix too large for GPU VRAM; "
                "using pin_memory for faster DMA transfers"
            )
    else:
        data_x = data.x

    all_run_results = []

    run_bar = tqdm(range(args.num_runs), desc="Runs", unit="run")
    for run in run_bar:
        run_bar.set_postfix(run=run)
        logger.info(f"{'=' * 40}  Run {run:02d}  {'=' * 40}")

        # Load checkpoints
        path_v2  = ckpt_path(args.ckpt_dir, args.dataset, args.task, "gatv2", run)
        path_gat = ckpt_path(args.ckpt_dir, args.dataset, args.task, "gat",   run)

        ckpt_v2  = load_checkpoint(path_v2,  device)
        ckpt_gat = load_checkpoint(path_gat, device)

        # AccGap: GATv2 test metric minus GAT test metric (stored in checkpoint)
        acc_gap = float(ckpt_v2["test_metric"]) - float(ckpt_gat["test_metric"])
        logger.info(
            f"AccGap = GATv2({ckpt_v2['test_metric']:.4f}) "
            f"- GAT({ckpt_gat['test_metric']:.4f}) = {acc_gap:.4f}"
        )

        # Rebuild and load models
        model_v2  = rebuild_model(ckpt_v2["args"],  in_channels, task_type)
        model_gat = rebuild_model(ckpt_gat["args"], in_channels, task_type)
        model_v2.load_state_dict(ckpt_v2["model_state"])
        model_gat.load_state_dict(ckpt_gat["model_state"])
        model_v2.to(device).eval()
        model_gat.to(device).eval()

        # Verify parameter parity between GAT and GATv2
        n_v2  = sum(p.numel() for p in model_v2.parameters())
        n_gat = sum(p.numel() for p in model_gat.parameters())
        if n_v2 != n_gat:
            logger.warning(
                f"Parameter count mismatch: GATv2={n_v2:,}  GAT={n_gat:,}  "
                f"(share_weights=True should equalise these)"
            )
        else:
            logger.info(f"Parameter parity confirmed: {n_v2:,} params each")

        run_record = {"run": run, "acc_gap": acc_gap, "by_K": {}}

        k_bar = tqdm(args.subsample_sizes, desc=f"  Run {run:02d} K", unit="K",
                     leave=False)
        for K in k_bar:
            k_bar.set_postfix(K=K)
            logger.info(f"  -- K={K} --")
            # Cap K at num_nodes (full-graph mode)
            K_actual = min(K, data.num_nodes)
            dd_vals = compute_dd_for_subgraph(
                model_v2=model_v2,
                model_gat=model_gat,
                data=data,
                data_x=data_x,
                K=K_actual,
                run=run,
                device=device,
                k_list=args.k_list,
                layer_spec=args.layer,
                pair_chunk=args.pair_chunk,
                neighbor_cap=args.neighbor_cap,
                num_hops=args.num_hops,
                logger=logger,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            run_record["by_K"][str(K)] = dd_vals
            k_bar.set_postfix(K=K, RD=f"{dd_vals.get('rd_dd', float('nan')):.3f}",
                              CQ=f"{dd_vals.get('cq_dd', float('nan')):.3f}")

        run_bar.set_postfix(run=run, acc_gap=f"{acc_gap:.4f}")
        all_run_results.append(run_record)

    # Compute per-K summaries across runs
    summaries = {}
    for K in args.subsample_sizes:
        K_str = str(K)
        rows = [r["by_K"][K_str] for r in all_run_results if K_str in r["by_K"]]
        if rows and isinstance(rows[0], dict) and "rd_dd" in rows[0]:
            summaries[f"K_{K}"] = _summarise(rows, args.k_list)
            s = summaries[f"K_{K}"]
            logger.info(
                f"K={K} SUMMARY: "
                f"RD={s['rd_dd_mean']:.4f}±{s['rd_dd_std']:.4f}  "
                f"CQ={s['cq_dd_mean']:.4f}±{s['cq_dd_std']:.4f}  "
                f"GG={s['gg_dd_mean']:.4f}±{s['gg_dd_std']:.4f}  "
                f"AEV={s['aev_dd_mean']:.4f}±{s['aev_dd_std']:.4f}"
            )

    # AccGap summary
    acc_gaps = [r["acc_gap"] for r in all_run_results]
    logger.info(
        f"AccGap: {np.mean(acc_gaps):.4f} ± {np.std(acc_gaps):.4f} "
        f"across {args.num_runs} runs"
    )

    # Save JSON
    output = {
        "dataset": args.dataset,
        "task":    args.task,
        "num_runs": args.num_runs,
        "subsample_sizes": args.subsample_sizes,
        "k_list": args.k_list,
        "layer": args.layer,
        "runs": all_run_results,
        "summary": summaries,
    }
    json_path = os.path.join(
        args.output_dir, f"{args.dataset}_{args.task}_dd_results.json"
    )
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved → {json_path}")

    # Save primary-K CSV (one row per run) — use largest subsample size as primary
    primary_K = str(max(args.subsample_sizes))
    dd_keys = ["rd_dd", "cq_dd", "gg_dd", "aev_dd"] + [
        f"tki_dd_{k}" for k in args.k_list
    ]
    csv_path = os.path.join(
        args.output_dir, f"{args.dataset}_{args.task}_dd_summary.csv"
    )
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["run", "acc_gap"] + dd_keys
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_run_results:
            row = {"run": r["run"], "acc_gap": r["acc_gap"]}
            by_k = r["by_K"].get(primary_K, {})
            for key in dd_keys:
                row[key] = by_k.get(key, float("nan"))
            writer.writerow(row)
    logger.info(f"Summary CSV saved → {csv_path}")
    logger.info("Stage 2 complete.")


if __name__ == "__main__":
    main()
