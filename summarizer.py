from pathlib import Path
import pandas as pd
import numpy as np

CSV_DIR = Path("./results/attention_tables")
OUT_FILE = Path("./results/statistical_summary.txt")
LAYERS = [0, 1]


def fmt(label: str, val: float) -> str:
    return f"    {label:<22}: {val:.6f}"


def base_stats(s: pd.Series) -> list[str]:
    s = s.dropna()
    return [
        fmt("mean",   s.mean()),
        fmt("std",    s.std()),
        fmt("median", s.median()),
        fmt("min",    s.min()),
        fmt("max",    s.max()),
    ]


def write_section(f, title: str, lines: list[str]):
    f.write(f"  {title}\n")
    for line in lines:
        f.write(line + "\n")
    f.write("\n")


with open(OUT_FILE, "w") as f:
    for layer in LAYERS:
        f.write(f"{'='*60}\n")
        f.write(f"  LAYER {layer}\n")
        f.write(f"{'='*60}\n\n")

        # Edges
        edf = pd.read_csv(CSV_DIR / f"edges_layer{layer}.csv")

        gat_cols   = [c for c in edf.columns if c.startswith("alpha_gat_h")]
        gatv2_cols = [c for c in edf.columns if c.startswith("alpha_gatv2_h")]

        # flatten all head columns into one series for summary
        alpha_gat_all   = edf[gat_cols].values.flatten()
        alpha_gatv2_all = edf[gatv2_cols].values.flatten()

        f.write("  [ EDGES ]\n\n")

        gat_s = pd.Series(alpha_gat_all).dropna()
        write_section(f, "alpha_gat (all heads)", base_stats(pd.Series(alpha_gat_all)))

        gatv2_s = pd.Series(alpha_gatv2_all).dropna()
        write_section(f, "alpha_gatv2 (all heads)", base_stats(pd.Series(alpha_gatv2_all)))

        ec = edf["edge_cosine"].dropna()
        write_section(f, "edge_cosine", base_stats(ec) + [
            fmt("frac > 0.9", (ec > 0.9).mean()),
            fmt("frac < 0.1", (ec < 0.1).mean()),
        ])

        # Nodes
        ndf = pd.read_csv(CSV_DIR / f"nodes_layer{layer}.csv")

        f.write("  [ NODES ]\n\n")

        nc = ndf["node_cos"].dropna()
        write_section(f, "node_cos", base_stats(nc) + [
            fmt("frac > 0.9", (nc > 0.9).mean()),
            fmt("frac < 0.1", (nc < 0.1).mean()),
        ])

        nj = ndf["node_jsd"].dropna()
        write_section(f, "node_jsd", base_stats(nj) + [
            fmt("frac > 0.35", (nj > 0.35).mean()),
        ])

        ns = ndf["node_spearman"].dropna()
        write_section(f, "node_spearman", base_stats(ns) + [
            fmt("frac > 0.9", (ns > 0.9).mean()),
            fmt("count (non-NaN)", len(ns)),
        ])
