"""
Stage 3 — RQ3: Unsupervised Proxy for DD.

Evaluates three pre-training proxies for DD computed purely from graph
structure and raw node features (no trained model needed):

  P1 — NFD  : Neighborhood Feature Diversity
  P2 — SHS  : Structural Heterophily Score = 1 - homophily
  P3 — DWCD : Degree-Weighted Clustering Deficit = E_v[deg(v)·(1−C_v)]
  P4 — Composite: LOO-fit linear combination of P1/P2/P3

Each proxy is evaluated on:
  (a) Spearman ρ(P, DD_best) — how well it approximates the best DD metric
  (b) Spearman ρ(P, acc_gap) — how well it predicts the target

If ρ(P, acc_gap) ≥ 0.7 for any proxy: print a threshold-based decision rule.

Outputs:
  rq3_proxy_comparison.csv  — ρ against DD and acc_gap for each proxy
  rq3_proxy_scatter.pdf     — scatter plots: proxy vs acc_gap and proxy vs DD_best

Usage:
  python -m analysis.rq3_proxies --analysis_dir .data/analysis \\
      [--dd_best rd_dd]
"""

import argparse
import math
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.plot_utils import (
    DD_COLS, DD_DISPLAY, set_style,
    dataset_color, dataset_label, annotate_points, theil_sen_line,
    filter_by_task, task_suffix,
)


# ---------------------------------------------------------------------------
# Proxy computation from structural_features.csv
# ---------------------------------------------------------------------------

def build_proxies(df_struct):
    """
    Compute P1 (NFD), P2 (SHS), P3 (DWCD) from structural_features.csv columns.
    DWCD = degree_mean * (1 - clustering_mean); uses already-aggregated stats.
    Returns df with one row per dataset, columns: dataset, p1, p2, p3.
    """
    df = df_struct.copy()

    df["p1"] = pd.to_numeric(df.get("nfd", float("nan")), errors="coerce")
    df["p2"] = pd.to_numeric(df.get("shs", float("nan")), errors="coerce")

    deg_mean = pd.to_numeric(df.get("degree_mean", float("nan")), errors="coerce")
    clust    = pd.to_numeric(df.get("clustering_mean", float("nan")), errors="coerce")
    df["p3"] = deg_mean * (1.0 - clust.fillna(0))  # if clustering is NaN, treat as 0

    return df[["dataset", "p1", "p2", "p3"]]


def _minmax_normalize(arr):
    arr = np.array(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) < 2:
        return arr
    mn, mx = finite.min(), finite.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = spearmanr(x[mask], y[mask])
    return float(res.statistic), float(res.pvalue)


def _loo_linear_fit(X, y):
    """
    LOO-fit a linear combination of X columns to predict y.
    Returns (predictions at each LOO hold-out, loo_r2).
    X: (n, d) array; y: (n,) array. Handles NaN by masking.
    """
    n = X.shape[0]
    preds = np.full(n, float("nan"))
    loo   = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr        = y[train_idx]
        # Require at least 2 training points and 1 feature without all-NaN
        mask_tr = np.all(np.isfinite(X_tr), axis=1) & np.isfinite(y_tr)
        if mask_tr.sum() < 2:
            continue
        reg = LinearRegression()
        reg.fit(X_tr[mask_tr], y_tr[mask_tr])
        if np.all(np.isfinite(X_te)):
            preds[test_idx[0]] = reg.predict(X_te)[0]

    valid = np.isfinite(preds) & np.isfinite(y)
    if valid.sum() < 2:
        return preds, float("nan")
    ss_res = np.sum((y[valid] - preds[valid]) ** 2)
    ss_tot = np.sum((y[valid] - y[valid].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return preds, float(r2)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_proxies(df_proxies, df_summary, primary_K, dd_best_col, output_dir, sfx=""):
    df_dd = df_summary[df_summary["K"] == primary_K].copy()
    df    = df_dd.merge(df_proxies, on="dataset", how="inner")
    n     = len(df)
    print(f"  Datasets: {df['dataset'].tolist()}  (n={n})")

    acc_gap  = df["acc_gap_mean"].values.astype(float)
    dd_best_col_mean = f"{dd_best_col}_mean"
    dd_vals  = df[dd_best_col_mean].values.astype(float) if dd_best_col_mean in df.columns \
               else np.full(n, float("nan"))

    # Normalize proxies
    p1_raw = df["p1"].values.astype(float)
    p2_raw = df["p2"].values.astype(float)
    p3_raw = df["p3"].values.astype(float)
    p1 = _minmax_normalize(p1_raw)
    p2 = _minmax_normalize(p2_raw)
    p3 = _minmax_normalize(p3_raw)

    # P4: LOO composite
    X = np.column_stack([p1, p2, p3])
    p4_preds, p4_r2 = _loo_linear_fit(X, acc_gap)

    proxies = {
        "P1_NFD":       p1,
        "P2_SHS":       p2,
        "P3_DWCD":      p3,
        "P4_Composite": p4_preds,
    }

    rows = []
    for pname, pvals in proxies.items():
        rho_dd,  _ = _spearman(pvals, dd_vals)
        rho_gap, _ = _spearman(pvals, acc_gap)
        rows.append({
            "proxy":            pname,
            "rho_vs_dd_best":   rho_dd,
            "rho_vs_acc_gap":   rho_gap,
            "dd_best_used":     dd_best_col,
            "n_datasets":       int(np.isfinite(pvals).sum()),
            "p4_loo_r2":        p4_r2 if pname == "P4_Composite" else float("nan"),
        })
        print(f"  {pname:16s}  ρ(P, DD_best)={rho_dd:.3f}  ρ(P, AccGap)={rho_gap:.3f}")

    df_out = pd.DataFrame(rows)
    comp_path = os.path.join(output_dir, f"rq3_proxy_comparison{sfx}.csv")
    df_out.to_csv(comp_path, index=False)
    print(f"  Proxy comparison   → {comp_path}")

    # Decision rule for any proxy that achieves ρ >= 0.7 with acc_gap
    for row in rows:
        if math.isfinite(row["rho_vs_acc_gap"]) and abs(row["rho_vs_acc_gap"]) >= 0.7:
            pname  = row["proxy"]
            pvals  = proxies[pname]
            # Threshold: value of proxy at the dataset closest to acc_gap = 0
            acc_abs = np.abs(acc_gap)
            min_idx = int(np.argmin(acc_abs))
            tau     = float(pvals[min_idx]) if math.isfinite(pvals[min_idx]) else float("nan")
            print(f"\n  >> Decision rule ({pname}): use GATv2 if P > {tau:.3f}  "
                  f"(threshold from {df['dataset'].iloc[min_idx]}, AccGap≈0)")

    return proxies, df, acc_gap, dd_vals


# ---------------------------------------------------------------------------
# Scatter plots
# ---------------------------------------------------------------------------

def proxy_scatter(proxies, df, acc_gap, dd_vals, dd_best_col, output_dir, sfx=""):
    set_style()
    pnames = list(proxies.keys())
    ncols  = 2
    nrows  = len(pnames)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.2))

    datasets = df["dataset"].tolist()
    colors   = [dataset_color(d) for d in datasets]

    for ri, (pname, pvals) in enumerate(proxies.items()):
        # Left: proxy vs acc_gap
        ax_l = axes[ri][0]
        ax_l.scatter(pvals, acc_gap, c=colors, s=60, zorder=3)
        annotate_points(ax_l, pvals, acc_gap, datasets, fontsize=7)
        theil_sen_line(ax_l, pvals, acc_gap)
        rho, _ = _spearman(pvals, acc_gap)
        ax_l.set_title(f"{pname} vs AccGap  (ρ={rho:.2f})" if math.isfinite(rho)
                       else f"{pname} vs AccGap  (ρ=nan)", fontsize=9)
        ax_l.set_xlabel(pname, fontsize=9)
        ax_l.set_ylabel("AccGap", fontsize=9)

        # Right: proxy vs DD_best
        ax_r = axes[ri][1]
        ax_r.scatter(pvals, dd_vals, c=colors, s=60, zorder=3)
        annotate_points(ax_r, pvals, dd_vals, datasets, fontsize=7)
        theil_sen_line(ax_r, pvals, dd_vals)
        rho2, _ = _spearman(pvals, dd_vals)
        ax_r.set_title(f"{pname} vs {DD_DISPLAY.get(dd_best_col, dd_best_col)}  "
                       f"(ρ={rho2:.2f})" if math.isfinite(rho2)
                       else f"{pname} vs {DD_DISPLAY.get(dd_best_col, dd_best_col)}  (ρ=nan)",
                       fontsize=9)
        ax_r.set_xlabel(pname, fontsize=9)
        ax_r.set_ylabel(DD_DISPLAY.get(dd_best_col, dd_best_col), fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"rq3_proxy_scatter{sfx}.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Proxy scatter      → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RQ3: Unsupervised proxy evaluation")
    p.add_argument("--analysis_dir", required=True,
                   help="Directory containing df_summary.csv, structural_features.csv")
    p.add_argument("--dd_best", default=None,
                   help="DD metric to use as 'best DD' target (default: auto-select "
                        "highest ρ with acc_gap from rq1_correlation_table.csv, "
                        "or 'rd_dd' if table is missing)")
    p.add_argument("--primary_K", type=int, default=None,
                   help="K value to use for analysis (default: largest available)")
    p.add_argument("--task", type=str, default="all",
                   help="Restrict to one task type: 'node_classification' (alias: 'node_clf'), "
                        "'link_prediction' (alias: 'link_pred'), or 'all' (default).")
    return p.parse_args()


def main():
    args = parse_args()
    summary_path = os.path.join(args.analysis_dir, "df_summary.csv")
    struct_path  = os.path.join(args.analysis_dir, "structural_features.csv")

    if not os.path.exists(summary_path):
        print(f"Run collect_results.py first — missing {summary_path}")
        return
    if not os.path.exists(struct_path):
        print(f"Run structural_features.py first — missing {struct_path}")
        return

    df_summary = pd.read_csv(summary_path)
    df_struct  = pd.read_csv(struct_path)

    df_summary = filter_by_task(df_summary, args.task)
    sfx        = task_suffix(args.task)
    kept_datasets = df_summary["dataset"].unique()
    df_struct = df_struct[df_struct["dataset"].isin(kept_datasets)]
    print(f"Task filter: '{args.task}'  →  datasets: {sorted(kept_datasets)}")

    K_values  = sorted(df_summary["K"].unique())
    primary_K = args.primary_K if args.primary_K else max(K_values)

    # Auto-select dd_best from the task-matched RQ1 correlation table if available
    dd_best_col = args.dd_best
    if dd_best_col is None:
        corr_path = os.path.join(args.analysis_dir, f"rq1_correlation_table{sfx}.csv")
        if not os.path.exists(corr_path):
            corr_path = os.path.join(args.analysis_dir, "rq1_correlation_table.csv")
        if os.path.exists(corr_path):
            df_corr = pd.read_csv(corr_path)
            df_corr_valid = df_corr[df_corr["rho"].notna()]
            if len(df_corr_valid):
                best_row = df_corr_valid.loc[df_corr_valid["rho"].abs().idxmax()]
                dd_best_col = best_row["dd_metric"]
                print(f"Auto-selected dd_best = '{dd_best_col}' (highest |ρ| with acc_gap)")
        if dd_best_col is None:
            dd_best_col = "rd_dd"
            print(f"Defaulting dd_best = '{dd_best_col}'")

    df_proxies = build_proxies(df_struct)

    print(f"\n--- RQ3: Proxy evaluation  (dd_best={dd_best_col}, K={primary_K}) ---")
    proxies, df, acc_gap, dd_vals = evaluate_proxies(
        df_proxies, df_summary, primary_K, dd_best_col, args.analysis_dir, sfx
    )

    print("\n--- Proxy scatter plots ---")
    proxy_scatter(proxies, df, acc_gap, dd_vals, dd_best_col, args.analysis_dir, sfx)


if __name__ == "__main__":
    main()
