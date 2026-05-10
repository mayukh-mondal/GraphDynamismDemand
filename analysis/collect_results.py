"""
Stage 3 — Data Aggregation.

Scans --dd_dir for all *_dd_results.json files (produced by dd/compute_dd.py),
flattens the nested structure into two tidy DataFrames, and saves them as CSV.
Also pulls SAGE test metrics from Stage 1 checkpoints to add a sage_gap column.

Outputs (written to --output_dir):
  df_runs.csv    — one row per (dataset, run, K)
  df_summary.csv — one row per (dataset, K), mean ± std aggregated across runs

Usage:
  python -m analysis.collect_results \\
      --dd_dir .data/dd_results \\
      --ckpt_dir .data/ckpts \\
      --output_dir .data/analysis
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd
import torch

DD_COLS = ["rd_dd", "cq_dd", "gg_dd", "aev_dd", "tki_dd_1", "tki_dd_3", "tki_dd_5"]


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_dd_json(path):
    with open(path) as f:
        return json.load(f)


def _flatten_runs(doc):
    """
    Flatten doc["runs"] into a list of row dicts with keys:
      dataset, task, run, K, acc_gap, rd_dd, cq_dd, gg_dd, aev_dd,
      tki_dd_1, tki_dd_3, tki_dd_5
    """
    rows = []
    dataset = doc["dataset"]
    task    = doc["task"]
    for run_entry in doc["runs"]:
        run     = run_entry["run"]
        acc_gap = run_entry["acc_gap"]
        for K_str, metrics in run_entry["by_K"].items():
            row = {
                "dataset": dataset,
                "task":    task,
                "run":     run,
                "K":       int(K_str),
                "acc_gap": acc_gap,
            }
            for col in DD_COLS:
                row[col] = metrics.get(col, float("nan"))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# SAGE gap extraction from Stage 1 checkpoints
# ---------------------------------------------------------------------------

def _load_sage_metrics(ckpt_dir, dataset, task, num_runs):
    """
    Read test_metric from sage_run{i:02d}_best.pt checkpoints.
    Returns dict {run: sage_test_metric} for available runs.
    """
    metrics = {}
    for run in range(num_runs):
        path = os.path.join(ckpt_dir, dataset, task, f"sage_run{run:02d}_best.pt")
        if not os.path.exists(path):
            continue
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            metrics[run] = float(ckpt.get("test_metric", float("nan")))
        except Exception:
            pass
    return metrics


def _load_gat_metrics(ckpt_dir, dataset, task, num_runs):
    """Read GAT (static attention) test metrics for sage_gap computation."""
    metrics = {}
    for run in range(num_runs):
        path = os.path.join(ckpt_dir, dataset, task, f"gat_run{run:02d}_best.pt")
        if not os.path.exists(path):
            continue
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            metrics[run] = float(ckpt.get("test_metric", float("nan")))
        except Exception:
            pass
    return metrics


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def collect(dd_dir, ckpt_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(dd_dir, "*_dd_results.json")))
    if not json_files:
        print(f"No *_dd_results.json files found in {dd_dir}")
        return

    all_rows = []
    for jf in json_files:
        doc  = load_dd_json(jf)
        rows = _flatten_runs(doc)

        # Attach sage_gap if checkpoints exist
        if ckpt_dir:
            num_runs   = doc["num_runs"]
            dataset    = doc["dataset"]
            task       = doc["task"]
            sage_m     = _load_sage_metrics(ckpt_dir, dataset, task, num_runs)
            gat_m      = _load_gat_metrics(ckpt_dir,  dataset, task, num_runs)
            for row in rows:
                run = row["run"]
                s   = sage_m.get(run, float("nan"))
                g   = gat_m.get(run, float("nan"))
                row["sage_gap"] = s - g if (math.isfinite(s) and math.isfinite(g)) else float("nan")
        else:
            for row in rows:
                row["sage_gap"] = float("nan")

        all_rows.extend(rows)
        nan_metrics = [c for c in DD_COLS if all(math.isnan(r[c]) for r in rows)]
        if nan_metrics:
            print(f"  [{doc['dataset']}] WARNING — all NaN for: {nan_metrics}")
        else:
            print(f"  [{doc['dataset']}] OK — {len(rows)} rows, runs={doc['num_runs']}, K={doc['subsample_sizes']}")

    df_runs = pd.DataFrame(all_rows)
    # Ensure numeric types
    for col in DD_COLS + ["acc_gap", "K", "run", "sage_gap"]:
        if col in df_runs.columns:
            df_runs[col] = pd.to_numeric(df_runs[col], errors="coerce")

    # --- df_summary: mean ± std per (dataset, K) across runs ---
    agg_dict = {col: ["mean", "std"] for col in DD_COLS + ["acc_gap", "sage_gap"]}
    df_summary = (
        df_runs
        .groupby(["dataset", "task", "K"])
        .agg(agg_dict)
        .reset_index()
    )
    # Flatten multi-level columns
    df_summary.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in df_summary.columns
    ]

    runs_path    = os.path.join(output_dir, "df_runs.csv")
    summary_path = os.path.join(output_dir, "df_summary.csv")
    df_runs.to_csv(runs_path,    index=False)
    df_summary.to_csv(summary_path, index=False)

    print(f"\ndf_runs    → {runs_path}  ({len(df_runs)} rows)")
    print(f"df_summary → {summary_path}  ({len(df_summary)} rows)")
    return df_runs, df_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Aggregate Stage 2 DD results into tidy CSVs")
    p.add_argument("--dd_dir",     required=True, help="Directory containing *_dd_results.json files")
    p.add_argument("--ckpt_dir",   default=None,  help="Stage 1 checkpoint root (for SAGE metrics)")
    p.add_argument("--output_dir", required=True, help="Where to write df_runs.csv and df_summary.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect(args.dd_dir, args.ckpt_dir, args.output_dir)
