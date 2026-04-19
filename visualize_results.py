from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = Path("./results/attention_tables")
FIG_DIR = Path("./figures")
LAYERS  = [0, 1]


def safe_lim(lo, hi):
    if lo == hi:
        return lo - 0.5, hi + 0.5
    return lo, hi


def xlim_of(arr):
    a = arr[np.isfinite(arr)]
    return safe_lim(float(a.min()), float(a.max()))


def ylim_of(arr):
    return xlim_of(arr)


def plot_hist(ax, vals, color, xlabel, title):
    v = vals[np.isfinite(vals)]
    lo, hi = xlim_of(v)
    ax.hist(v, bins=100, color=color, edgecolor="none", alpha=0.85, density=True)
    ax.axvline(v.mean(),      color="#dc2626", lw=1.5, label=f"mean={v.mean():.4f}")
    ax.axvline(np.median(v),  color="#16a34a", lw=1.5, ls="--", label=f"median={np.median(v):.4f}")
    ax.set_xlim(lo, hi)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)


def plot_deg_scatter(ax, deg, metric, color, ylabel, title):
    valid   = np.isfinite(metric)
    deg_v   = deg[valid].astype(float)
    met_v   = metric[valid]

    rng    = np.random.default_rng(42)
    idx    = rng.choice(len(deg_v), min(50_000, len(deg_v)), replace=False)

    log_deg = np.log1p(deg_v)
    bins    = np.unique(np.percentile(log_deg, np.linspace(0, 100, 21)))
    bin_idx = np.clip(np.digitize(log_deg, bins) - 1, 0, len(bins) - 2)

    centers, means, stds = [], [], []
    for b in range(len(bins) - 1):
        mask = bin_idx == b
        if mask.sum() > 5:
            centers.append(np.expm1(bins[b]))
            means.append(met_v[mask].mean())
            stds.append(met_v[mask].std())
    centers, means, stds = np.array(centers), np.array(means), np.array(stds)

    ax.scatter(deg_v[idx], met_v[idx], alpha=0.05, s=2, color="#94a3b8", rasterized=True)
    ax.plot(centers, means, color=color, lw=2, label="Bin mean")
    ax.fill_between(centers, means - stds, means + stds, alpha=0.2, color=color)
    ax.set_xscale("log")
    ax.set_ylim(*ylim_of(met_v))
    ax.set_xlabel("Node degree (log scale)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)


def plot_head_scatter(ax, gat_h, gatv2_h, h):
    lo = min(float(gat_h.min()), float(gatv2_h.min()))
    hi = max(float(gat_h.max()), float(gatv2_h.max()))
    lo, hi = safe_lim(lo, hi)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(gat_h), min(50_000, len(gat_h)), replace=False)

    ax.scatter(gat_h[idx], gatv2_h[idx], alpha=0.05, s=1, color="#2563eb", rasterized=True)
    ax.plot([lo, hi], [lo, hi], color="#dc2626", lw=1, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("alpha_gat", fontsize=9)
    ax.set_ylabel("alpha_gatv2", fontsize=9)
    ax.set_title(f"Head {h}", fontsize=10)


def export_layer(layer: int, out_dir: Path):
    edf = pd.read_csv(CSV_DIR / f"edges_layer{layer}.csv")
    ndf = pd.read_csv(CSV_DIR / f"nodes_layer{layer}.csv")

    gat_cols   = [c for c in edf.columns if c.startswith("alpha_gat_h")]
    gatv2_cols = [c for c in edf.columns if c.startswith("alpha_gatv2_h")]
    H = len(gat_cols)

    alpha_gat_all   = edf[gat_cols].values.flatten()
    alpha_gatv2_all = edf[gatv2_cols].values.flatten()
    edge_cos        = edf["edge_cosine"].values
    deg             = ndf["deg"].values.astype(float)
    node_cos        = ndf["node_cos"].values
    node_jsd        = ndf["node_jsd"].values
    node_spearman   = ndf["node_spearman"].values

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.suptitle(f"Distributions · Layer {layer}", fontsize=13, fontweight="bold")

    plot_hist(axes[0, 0], edge_cos,        "#2563eb", "Edge Cosine Similarity", "Edge Cosine")
    plot_hist(axes[0, 1], alpha_gat_all,   "#7c3aed", "Attention Weight",       "alpha_gat (all heads)")
    plot_hist(axes[1, 0], alpha_gatv2_all, "#0891b2", "Attention Weight",       "alpha_gatv2 (all heads)")
    plot_hist(axes[1, 1], node_cos,        "#7c3aed", "Node Cosine Similarity", "Node Cosine")
    plot_hist(axes[2, 0], node_jsd,        "#0891b2", "JSD (nats)",             "Node JSD")
    plot_hist(axes[2, 1], node_spearman,   "#b45309", "Spearman ρ",             "Node Spearman ρ")

    fig.tight_layout()
    fig.savefig(out_dir / "distributions.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Degree vs Node Metrics · Layer {layer}", fontsize=13, fontweight="bold")

    plot_deg_scatter(axes[0], deg, node_cos,      "#7c3aed", "Node Cosine",  "Degree vs Node Cosine")
    plot_deg_scatter(axes[1], deg, node_jsd,      "#0891b2", "Node JSD",     "Degree vs Node JSD")
    plot_deg_scatter(axes[2], deg, node_spearman, "#b45309", "Spearman ρ",   "Degree vs Node Spearman ρ")

    fig.tight_layout()
    fig.savefig(out_dir / "degree_vs_node_metrics.png", dpi=150)
    plt.close(fig)

    ncols = 4
    nrows = (H + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = np.array(axes).flatten()
    fig.suptitle(f"alpha_gat vs alpha_gatv2 per Head · Layer {layer}", fontsize=13, fontweight="bold")

    for h in range(H):
        plot_head_scatter(axes[h], edf[gat_cols[h]].values, edf[gatv2_cols[h]].values, h)
    for h in range(H, len(axes)):
        axes[h].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "head_scatter.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Per-head Attention Distributions · Layer {layer}", fontsize=13, fontweight="bold")

    labels     = [f"h{h}" for h in range(H)]
    gat_data   = [edf[c].values for c in gat_cols]
    gatv2_data = [edf[c].values for c in gatv2_cols]

    for ax, data, color, median_color, title in [
        (axes[0], gat_data,   "#93c5fd", "#1d4ed8", "alpha_gat per Head"),
        (axes[1], gatv2_data, "#6ee7b7", "#065f46", "alpha_gatv2 per Head"),
    ]:
        ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=color, alpha=0.7),
                   medianprops=dict(color=median_color, lw=2))
        ax.set_ylim(*ylim_of(np.concatenate(data)))
        ax.set_xlabel("Head", fontsize=10)
        ax.set_ylabel("Attention Weight", fontsize=10)
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    fig.savefig(out_dir / "head_boxplots.png", dpi=150)
    plt.close(fig)


for layer in LAYERS:
    out_dir = FIG_DIR / f"layer{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_layer(layer, out_dir)
