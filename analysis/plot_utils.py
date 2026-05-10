"""
Shared matplotlib style for all Stage 3 analysis figures.
Import set_style() at the top of any analysis script before creating figures.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Consistent dataset display ordering (roughly by expected DD, low→high)
DATASET_ORDER = [
    "ogbl-collab",
    "ogbl-citation2",
    "ogbn-arxiv",
    "ogbn-mag",
    "ogbn-products",
    "ogbn-proteins",
]

# Color palette: one color per dataset, colorblind-friendly
DATASET_COLORS = {
    "ogbn-arxiv":    "#4C72B0",
    "ogbn-products": "#DD8452",
    "ogbn-proteins": "#55A868",
    "ogbn-mag":      "#C44E52",
    "ogbl-collab":   "#8172B2",
    "ogbl-citation2":"#937860",
}

# Short display names for scatter plot labels
DATASET_LABELS = {
    "ogbn-arxiv":    "arxiv",
    "ogbn-products": "products",
    "ogbn-proteins": "proteins",
    "ogbn-mag":      "mag",
    "ogbl-collab":   "collab",
    "ogbl-citation2":"cit2",
}

# Human-readable DD metric names
DD_DISPLAY = {
    "rd_dd":    "RD-DD",
    "cq_dd":    "CQ-DD",
    "gg_dd":    "DAA-DD",   # Dynamic Attention Advantage = CQ(GATv2) - CQ(GAT)
    "aev_dd":   "EL-DD",    # Entropy Lift = E[|H_norm_v2 - H_norm_gat|]
    "tki_dd_1": "TKI-DD (k=1)",
    "tki_dd_3": "TKI-DD (k=3)",
    "tki_dd_5": "TKI-DD (k=5)",
}

DD_COLS = ["rd_dd", "cq_dd", "gg_dd", "aev_dd", "tki_dd_1", "tki_dd_3", "tki_dd_5"]


TASK_ALIASES = {
    "node_classification": "node_classification",
    "node_clf":            "node_classification",
    "link_prediction":     "link_prediction",
    "link_pred":           "link_prediction",
    "all":                 "all",
}


def filter_by_task(df, task):
    """
    Restrict a DataFrame (must have a 'task' column) to rows matching `task`.
    task="all" returns the full DataFrame unchanged.
    Accepts shorthand aliases: "node_clf", "link_pred".
    """
    canonical = TASK_ALIASES.get(task, task)
    if canonical == "all":
        return df
    if "task" not in df.columns:
        raise ValueError("DataFrame has no 'task' column — cannot filter by task")
    filtered = df[df["task"] == canonical]
    if filtered.empty:
        available = df["task"].unique().tolist()
        raise ValueError(
            f"No rows with task='{canonical}'. Available tasks: {available}"
        )
    return filtered


def task_suffix(task):
    """Return a short filename suffix for task-filtered outputs, e.g. '_ncf'."""
    canonical = TASK_ALIASES.get(task, task)
    return {"node_classification": "_ncf", "link_prediction": "_lp"}.get(canonical, "")


def set_style():
    """Apply project-wide matplotlib style."""
    matplotlib.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     "--",
    })


def dataset_color(name):
    return DATASET_COLORS.get(name, "#888888")


def dataset_label(name):
    return DATASET_LABELS.get(name, name)


def annotate_points(ax, xs, ys, labels, colors=None, fontsize=8, offset=(4, 4)):
    """Add dataset name annotations next to scatter points."""
    for x, y, lbl in zip(xs, ys, labels):
        c = dataset_color(lbl) if colors is None else "black"
        ax.annotate(
            dataset_label(lbl),
            xy=(x, y),
            xytext=(offset[0], offset[1]),
            textcoords="offset points",
            fontsize=fontsize,
            color=c,
            ha="left",
        )


def theil_sen_line(ax, x, y, color="gray", lw=1.2, alpha=0.6, n_points=100):
    """Overlay a Theil-Sen robust regression line on ax."""
    from scipy.stats import theilslopes
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return
    res = theilslopes(y[mask], x[mask])
    slope, intercept = res.slope, res.intercept
    xl = np.linspace(x[mask].min(), x[mask].max(), n_points)
    ax.plot(xl, intercept + slope * xl, color=color, lw=lw, alpha=alpha, ls="--")
