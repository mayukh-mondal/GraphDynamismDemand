"""
Stage 3 — RQ1: DD vs Accuracy Gap.

Reads df_runs.csv and df_summary.csv (from collect_results.py) and produces:

  rq1_correlation_table.csv — Spearman ρ(DD, acc_gap) for each DD definition,
                              with LOO confidence intervals
  rq1_scatter.pdf           — 2×4 subplot grid: DD definition vs acc_gap,
                              one scatter point per dataset (labeled)
  rq1_convergence.pdf       — DD(K) vs K per dataset, one line per DD definition
  rq1_run_level.csv         — within-dataset ρ(DD_k, acc_gap_k) across runs

All analyses operate on the largest available K (most data).

Usage:
  python -m analysis.rq1_dd_accgap --analysis_dir .data/analysis
"""

import argparse
import math
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import theilslopes  # noqa: used via plot_utils

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.plot_utils import (
    DD_COLS, DD_DISPLAY, set_style,
    dataset_color, dataset_label, annotate_points, theil_sen_line,
    filter_by_task, task_suffix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spearman(x, y):
    """Spearman ρ and p-value, handling NaN and n<2."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = spearmanr(x[mask], y[mask])
    return float(res.statistic), float(res.pvalue)


def _loo_spearman(x, y):
    """
    Leave-one-out Spearman ρ: for each i, compute ρ on all indices except i.
    Returns array of ρ values (length = n).
    """
    n = len(x)
    rhos = []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        xi  = np.array([x[j] for j in idx])
        yi  = np.array([y[j] for j in idx])
        r, _ = _spearman(xi, yi)
        rhos.append(r)
    return np.array(rhos)


# ---------------------------------------------------------------------------
# 3a. Cross-dataset correlation table
# ---------------------------------------------------------------------------

def correlation_table(df_summary, primary_K, output_dir, sfx=""):
    df = df_summary[df_summary["K"] == primary_K].copy()
    datasets = df["dataset"].tolist()
    acc_gap  = df["acc_gap_mean"].values

    rows = []
    for col in DD_COLS:
        mean_col = f"{col}_mean"
        if mean_col not in df.columns:
            continue
        dd_vals = df[mean_col].values
        rho, pval = _spearman(dd_vals, acc_gap)
        loo_rhos  = _loo_spearman(dd_vals, acc_gap)
        valid_loo = loo_rhos[np.isfinite(loo_rhos)]
        rows.append({
            "dd_metric":      col,
            "dd_display":     DD_DISPLAY.get(col, col),
            "rho":            rho,
            "pvalue":         pval,
            "loo_rho_mean":   float(np.mean(valid_loo)) if len(valid_loo) else float("nan"),
            "loo_rho_std":    float(np.std(valid_loo))  if len(valid_loo) else float("nan"),
            "loo_rho_min":    float(np.min(valid_loo))  if len(valid_loo) else float("nan"),
            "loo_rho_max":    float(np.max(valid_loo))  if len(valid_loo) else float("nan"),
            "sign_positive":  rho > 0 if math.isfinite(rho) else None,
            "n_datasets":     int(np.isfinite(dd_vals).sum()),
        })

    df_corr = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, f"rq1_correlation_table{sfx}.csv")
    df_corr.to_csv(out_path, index=False)
    print(f"Correlation table → {out_path}")
    print(df_corr[["dd_display", "rho", "loo_rho_mean", "loo_rho_std", "n_datasets"]].to_string(index=False))
    return df_corr


# ---------------------------------------------------------------------------
# 3b. Scatter plots: DD vs acc_gap (one subplot per DD definition)
# ---------------------------------------------------------------------------

def scatter_plots(df_summary, primary_K, output_dir, sfx=""):
    set_style()
    df = df_summary[df_summary["K"] == primary_K].copy()
    datasets = df["dataset"].tolist()

    plot_cols = [c for c in DD_COLS if f"{c}_mean" in df.columns]
    ncols = 4
    nrows = math.ceil(len(plot_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
    axes = np.array(axes).flatten()

    for ax_idx, col in enumerate(plot_cols):
        ax = axes[ax_idx]
        mean_col = f"{col}_mean"
        std_col  = f"{col}_std"

        dd_vals  = df[mean_col].values
        acc_vals = df["acc_gap_mean"].values

        acc_std = df.get("acc_gap_std", pd.Series([0.0] * len(df))).values
        dd_std  = df[std_col].values if std_col in df.columns else np.zeros_like(dd_vals)

        colors = [dataset_color(d) for d in datasets]
        ax.scatter(dd_vals, acc_vals, c=colors, s=60, zorder=3)
        ax.errorbar(dd_vals, acc_vals,
                    xerr=dd_std, yerr=acc_std,
                    fmt="none", ecolor="gray", alpha=0.4, zorder=2)

        annotate_points(ax, dd_vals, acc_vals, datasets, fontsize=7)
        theil_sen_line(ax, dd_vals, acc_vals)

        rho, _ = _spearman(dd_vals, acc_vals)
        rho_str = f"ρ={rho:.2f}" if math.isfinite(rho) else "ρ=nan"
        ax.set_title(f"{DD_DISPLAY.get(col, col)}\n{rho_str}", fontsize=10)
        ax.set_xlabel(DD_DISPLAY.get(col, col), fontsize=9)
        ax.set_ylabel("AccGap (GATv2 − GAT)", fontsize=9)

    # Hide unused axes
    for i in range(len(plot_cols), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"DD vs AccGap  (K={primary_K})", fontsize=12, y=1.01)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"rq1_scatter{sfx}.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Scatter plots      → {out_path}")


# ---------------------------------------------------------------------------
# 3c. Convergence: DD(K) vs K per dataset
# ---------------------------------------------------------------------------

def convergence_plots(df_runs, output_dir, sfx=""):
    set_style()
    datasets = df_runs["dataset"].unique()
    K_vals   = sorted(df_runs["K"].unique())

    if len(K_vals) < 2:
        print("Only one K value — skipping convergence plots.")
        return

    # Mean DD per (dataset, K) across runs
    agg = df_runs.groupby(["dataset", "K"])[DD_COLS].mean().reset_index()

    plot_cols = [c for c in DD_COLS if not agg[c].isna().all()]
    ncols = min(len(datasets), 3)
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.2),
                             squeeze=False)
    axes = axes.flatten()

    linestyles = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,2))]
    for ax_idx, ds in enumerate(sorted(datasets)):
        ax    = axes[ax_idx]
        sub   = agg[agg["dataset"] == ds].sort_values("K")
        for li, col in enumerate(plot_cols):
            ax.plot(sub["K"], sub[col],
                    marker="o", ls=linestyles[li % len(linestyles)],
                    label=DD_DISPLAY.get(col, col), lw=1.5)
        ax.set_title(dataset_label(ds), fontsize=10)
        ax.set_xlabel("K (subsample seeds)", fontsize=9)
        ax.set_ylabel("DD value", fontsize=9)
        if ax_idx == 0:
            ax.legend(fontsize=7, ncol=2)

    for i in range(len(datasets), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("DD convergence: DD(K) vs K", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"rq1_convergence{sfx}.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Convergence plots  → {out_path}")

    # Convergence table: min K at which |DD(K) - DD(K/2)| < 0.01
    conv_rows = []
    for ds in sorted(datasets):
        sub = agg[agg["dataset"] == ds].sort_values("K")
        for col in plot_cols:
            vals = sub[col].values
            ks   = sub["K"].values
            converged_at = None
            for i in range(1, len(ks)):
                if abs(vals[i] - vals[i-1]) < 0.01:
                    converged_at = int(ks[i])
                    break
            conv_rows.append({"dataset": ds, "dd_metric": col, "converged_at_K": converged_at})

    df_conv = pd.DataFrame(conv_rows)
    conv_path = os.path.join(output_dir, f"rq1_convergence_table{sfx}.csv")
    df_conv.to_csv(conv_path, index=False)
    print(f"Convergence table  → {conv_path}")


# ---------------------------------------------------------------------------
# 3c. Run-level within-dataset correlation
# ---------------------------------------------------------------------------

def run_level_correlation(df_runs, primary_K, output_dir, sfx=""):
    df = df_runs[df_runs["K"] == primary_K].copy()
    datasets = df["dataset"].unique()
    rows = []
    for ds in sorted(datasets):
        sub = df[df["dataset"] == ds]
        acc = sub["acc_gap"].values
        for col in DD_COLS:
            if col not in sub.columns:
                continue
            dd_vals = sub[col].values
            rho, _ = _spearman(dd_vals, acc)
            rows.append({"dataset": ds, "dd_metric": col,
                         "rho_run_level": rho, "n_runs": int(np.isfinite(dd_vals).sum())})

    df_rl = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, f"rq1_run_level{sfx}.csv")
    df_rl.to_csv(out_path, index=False)
    print(f"Run-level corr     → {out_path}")
    return df_rl


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RQ1: DD vs AccGap correlation analysis")
    p.add_argument("--analysis_dir", required=True,
                   help="Directory containing df_runs.csv and df_summary.csv")
    p.add_argument("--primary_K", type=int, default=None,
                   help="K value to use for cross-dataset analysis (default: largest K)")
    p.add_argument("--task", type=str, default="all",
                   help="Restrict analysis to one task type: 'node_classification' "
                        "(alias: 'node_clf'), 'link_prediction' (alias: 'link_pred'), "
                        "or 'all' (default). Cross-dataset comparisons are only valid "
                        "within a single task type.")
    return p.parse_args()


def main():
    args = parse_args()
    runs_path    = os.path.join(args.analysis_dir, "df_runs.csv")
    summary_path = os.path.join(args.analysis_dir, "df_summary.csv")

    if not os.path.exists(runs_path) or not os.path.exists(summary_path):
        print(f"Run collect_results.py first — missing {runs_path} or {summary_path}")
        return

    df_runs    = pd.read_csv(runs_path)
    df_summary = pd.read_csv(summary_path)

    # Task filter
    df_runs    = filter_by_task(df_runs,    args.task)
    df_summary = filter_by_task(df_summary, args.task)
    sfx        = task_suffix(args.task)
    print(f"Task filter: '{args.task}'  →  suffix '{sfx}'")

    # Determine primary K
    K_values = sorted(df_runs["K"].unique())
    primary_K = args.primary_K if args.primary_K else max(K_values)
    print(f"Available K values: {K_values}  →  using K={primary_K} for cross-dataset analysis")
    print(f"Datasets in analysis: {sorted(df_runs['dataset'].unique())}")

    os.makedirs(args.analysis_dir, exist_ok=True)

    print("\n--- 3a. Cross-dataset Spearman ρ ---")
    correlation_table(df_summary, primary_K, args.analysis_dir, sfx)

    print("\n--- 3b. Scatter plots ---")
    scatter_plots(df_summary, primary_K, args.analysis_dir, sfx)

    print("\n--- 3c. Convergence ---")
    convergence_plots(df_runs, args.analysis_dir, sfx)

    print("\n--- 3d. Run-level correlation ---")
    run_level_correlation(df_runs, primary_K, args.analysis_dir, sfx)


if __name__ == "__main__":
    main()
