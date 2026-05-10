"""
Stage 3 — RQ2: Structural Correlates of DD.

Correlates graph-theoretic structural features (from structural_features.py)
with DD metric values (from collect_results.py) across datasets.

Outputs:
  rq2_heatmap.pdf         — feature × DD Spearman ρ heatmap
  rq2_correlation.csv     — full ρ table
  rq2_loo_robustness.csv  — LOO ρ mean/range per (feature, DD) pair
  rq2_hypothesis.csv      — directional hypothesis test results (H1-H4)

Usage:
  python -m analysis.rq2_structural --analysis_dir .data/analysis
"""

import argparse
import math
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.plot_utils import DD_COLS, DD_DISPLAY, set_style, filter_by_task, task_suffix


# ---------------------------------------------------------------------------
# Structural feature metadata
# ---------------------------------------------------------------------------

STRUCT_FEATURES = [
    ("degree_mean",    "Mean degree",              "tier1"),
    ("degree_std",     "Degree std",               "tier2"),
    ("degree_entropy", "Degree entropy",           "tier2"),
    ("homophily",      "Homophily",                "tier1"),
    ("clustering_mean","Clustering coeff",         "tier2"),
    ("assortativity",  "Degree assortativity",     "tier2"),
    ("nfd",            "NFD",                      "tier3"),
    ("shs",            "Heterophily (SHS)",        "tier1"),
    ("spectral_gap",   "Spectral gap (λ₂ approx)","tier2"),
]

# Pre-registered directional hypotheses (RQ2, §4.3 in plan)
HYPOTHESES = [
    ("degree_mean",    "rd_dd", ">", "H1: mean degree → higher DD"),
    ("homophily",      "rd_dd", "<", "H2: homophily → lower DD"),
    ("clustering_mean","rd_dd", ">", "H3: clustering → higher DD"),
    ("nfd",            "rd_dd", ">", "H4: NFD → higher DD"),
]


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


def _loo_spearman(x, y):
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
# 5a. Correlation heatmap
# ---------------------------------------------------------------------------

def correlation_heatmap(df_struct, df_summary, primary_K, output_dir, sfx=""):
    df_dd = df_summary[df_summary["K"] == primary_K].copy()
    # Merge on dataset
    df = df_dd.merge(df_struct, on="dataset", how="inner")
    n  = len(df)
    print(f"  Datasets in merged table: {df['dataset'].tolist()}  (n={n})")

    feat_names   = [f[0] for f in STRUCT_FEATURES if f[0] in df.columns]
    feat_display = {f[0]: f[1] for f in STRUCT_FEATURES}

    dd_cols_avail = [c for c in DD_COLS if f"{c}_mean" in df.columns]

    rho_matrix = np.full((len(feat_names), len(dd_cols_avail)), float("nan"))
    for fi, feat in enumerate(feat_names):
        x = df[feat].values.astype(float)
        for di, col in enumerate(dd_cols_avail):
            y = df[f"{col}_mean"].values.astype(float)
            rho, _ = _spearman(x, y)
            rho_matrix[fi, di] = rho

    # Save CSV
    df_corr = pd.DataFrame(
        rho_matrix,
        index=[feat_display.get(f, f) for f in feat_names],
        columns=[DD_DISPLAY.get(c, c) for c in dd_cols_avail],
    )
    corr_path = os.path.join(output_dir, f"rq2_correlation{sfx}.csv")
    df_corr.to_csv(corr_path)
    print(f"  Correlation table  → {corr_path}")
    print(df_corr.to_string())

    # Heatmap
    set_style()
    fig, ax = plt.subplots(figsize=(max(5, len(dd_cols_avail) * 1.2),
                                    max(4, len(feat_names) * 0.7)))
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
    im   = ax.imshow(rho_matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)

    ax.set_xticks(range(len(dd_cols_avail)))
    ax.set_xticklabels([DD_DISPLAY.get(c, c) for c in dd_cols_avail],
                       rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels([feat_display.get(f, f) for f in feat_names], fontsize=9)

    # Annotate cells
    for fi in range(len(feat_names)):
        for di in range(len(dd_cols_avail)):
            v = rho_matrix[fi, di]
            txt = f"{v:.2f}" if math.isfinite(v) else "—"
            col = "white" if abs(v) > 0.6 else "black"
            ax.text(di, fi, txt, ha="center", va="center", fontsize=8, color=col)

    ax.set_title(f"Structural Feature × DD Spearman ρ  (K={primary_K}, n={n} datasets)",
                 fontsize=11)
    fig.tight_layout()
    hmap_path = os.path.join(output_dir, f"rq2_heatmap{sfx}.pdf")
    fig.savefig(hmap_path)
    plt.close(fig)
    print(f"  Heatmap            → {hmap_path}")

    return rho_matrix, feat_names, dd_cols_avail, df


# ---------------------------------------------------------------------------
# 5c. LOO robustness
# ---------------------------------------------------------------------------

def loo_robustness(df_struct, df_summary, primary_K, feat_names, dd_cols_avail, df_merged, output_dir, sfx=""):
    rows = []
    for feat in feat_names:
        x = df_merged[feat].values.astype(float)
        for col in dd_cols_avail:
            y = df_merged[f"{col}_mean"].values.astype(float)
            loo_rhos = _loo_spearman(x, y)
            valid    = loo_rhos[np.isfinite(loo_rhos)]
            rho, _   = _spearman(x, y)
            rows.append({
                "feature":       feat,
                "dd_metric":     col,
                "rho_full":      rho,
                "loo_rho_mean":  float(np.mean(valid)) if len(valid) else float("nan"),
                "loo_rho_std":   float(np.std(valid))  if len(valid) else float("nan"),
                "loo_rho_min":   float(np.min(valid))  if len(valid) else float("nan"),
                "loo_rho_max":   float(np.max(valid))  if len(valid) else float("nan"),
                "n_datasets":    int(np.isfinite(x).sum()),
            })
    df_loo = pd.DataFrame(rows)
    loo_path = os.path.join(output_dir, f"rq2_loo_robustness{sfx}.csv")
    df_loo.to_csv(loo_path, index=False)
    print(f"  LOO robustness     → {loo_path}")

    # Highlight most robust correlates (high mean ρ, low range)
    df_loo["loo_rho_range"] = df_loo["loo_rho_max"] - df_loo["loo_rho_min"]
    robust = df_loo[df_loo["loo_rho_mean"].abs() > 0.5].sort_values(
        "loo_rho_mean", ascending=False, key=abs
    )
    if len(robust):
        print("  Robust correlates (|LOO ρ mean| > 0.5):")
        print(robust[["feature", "dd_metric", "loo_rho_mean", "loo_rho_range"]].to_string(index=False))


# ---------------------------------------------------------------------------
# 5b. Hypothesis-driven direction checks
# ---------------------------------------------------------------------------

def hypothesis_checks(df_merged, output_dir, sfx=""):
    rows = []
    for feat, col, direction, label in HYPOTHESES:
        mean_col = f"{col}_mean"
        if feat not in df_merged.columns or mean_col not in df_merged.columns:
            rows.append({"hypothesis": label, "feature": feat, "dd_metric": col,
                         "rho": float("nan"), "direction": direction, "confirmed": None})
            continue
        x = df_merged[feat].values.astype(float)
        y = df_merged[mean_col].values.astype(float)
        rho, _ = _spearman(x, y)
        confirmed = None
        if math.isfinite(rho):
            confirmed = (rho > 0) if direction == ">" else (rho < 0)
        rows.append({
            "hypothesis": label,
            "feature":    feat,
            "dd_metric":  col,
            "rho":        rho,
            "direction":  direction,
            "confirmed":  confirmed,
        })
        status = "✓" if confirmed else ("✗" if confirmed is False else "?")
        print(f"  {status}  {label}: ρ={rho:.3f}" if math.isfinite(rho) else f"  ?  {label}: ρ=nan")

    df_hyp = pd.DataFrame(rows)
    hyp_path = os.path.join(output_dir, f"rq2_hypothesis{sfx}.csv")
    df_hyp.to_csv(hyp_path, index=False)
    print(f"  Hypothesis results → {hyp_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RQ2: Structural Correlates of DD")
    p.add_argument("--analysis_dir", required=True,
                   help="Directory containing df_summary.csv and structural_features.csv")
    p.add_argument("--primary_K", type=int, default=None,
                   help="K value to use for cross-dataset analysis (default: largest K)")
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
    # Restrict structural features to datasets that remain after task filtering
    kept_datasets = df_summary["dataset"].unique()
    df_struct = df_struct[df_struct["dataset"].isin(kept_datasets)]
    print(f"Task filter: '{args.task}'  →  datasets: {sorted(kept_datasets)}")

    K_values  = sorted(df_summary["K"].unique())
    primary_K = args.primary_K if args.primary_K else max(K_values)
    print(f"Using K={primary_K} for RQ2 analysis")

    print("\n--- 5a. Correlation heatmap ---")
    rho_matrix, feat_names, dd_cols_avail, df_merged = correlation_heatmap(
        df_struct, df_summary, primary_K, args.analysis_dir, sfx
    )

    print("\n--- 5b. Hypothesis checks ---")
    hypothesis_checks(df_merged, args.analysis_dir, sfx)

    print("\n--- 5c. LOO robustness ---")
    loo_robustness(
        df_struct, df_summary, primary_K,
        feat_names, dd_cols_avail, df_merged, args.analysis_dir, sfx
    )


if __name__ == "__main__":
    main()
