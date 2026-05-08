"""
summarize_dd.py — parse dd_proteins.pt and write a structured summary.

Outputs
-------
dd_results/dd_summary.txt   Human-readable report (matches summarizer.py style)
dd_results/dd_nodes.csv     Per-node table: node_id, degree, rho_bar, dd
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="dd_results/dd_proteins.pt")
    p.add_argument("--out_txt", default="dd_results/dd_summary.txt")
    p.add_argument("--out_csv", default="dd_results/dd_nodes.csv")
    p.add_argument("--top_hubs", type=int, default=10,
                   help="Number of highest-degree nodes to spotlight")
    return p.parse_args()


def fmt(label: str, val: float, width: int = 24) -> str:
    return f"    {label:<{width}}: {val:.6f}"


def dist_stats(arr: np.ndarray, label: str) -> list[str]:
    lines = [f"  [ {label} ]"]
    lines += [
        fmt("count (valid)",   len(arr)),
        fmt("mean",            arr.mean()),
        fmt("std",             arr.std()),
        fmt("min",             arr.min()),
        fmt("p5",              np.percentile(arr, 5)),
        fmt("p25",             np.percentile(arr, 25)),
        fmt("median",          np.median(arr)),
        fmt("p75",             np.percentile(arr, 75)),
        fmt("p95",             np.percentile(arr, 95)),
        fmt("max",             arr.max()),
    ]
    return lines


def degree_band_stats(dd: np.ndarray, deg: np.ndarray) -> list[str]:
    bands = [
        ("low    (3–5)",    (deg >= 3)  & (deg <= 5)),
        ("medium (6–20)",   (deg >= 6)  & (deg <= 20)),
        ("high   (21–100)", (deg >= 21) & (deg <= 100)),
        ("hub    (>100)",   (deg > 100)),
    ]
    lines = ["  [ Degree-stratified DD_proxy ]"]
    lines.append(f"    {'Band':<22}  {'nodes':>8}  {'mean dd':>10}  {'median dd':>10}  {'mean deg':>10}")
    lines.append("    " + "-" * 64)
    for name, mask in bands:
        if mask.sum() == 0:
            lines.append(f"    {name:<22}  {'—':>8}")
            continue
        lines.append(
            f"    {name:<22}  {mask.sum():>8,}  "
            f"{dd[mask].mean():>10.4f}  {np.median(dd[mask]):>10.4f}  "
            f"{deg[mask].mean():>10.1f}"
        )
    return lines


def hub_spotlight(dd: np.ndarray, rho: np.ndarray,
                  deg: np.ndarray, top_k: int) -> list[str]:
    idx    = np.argsort(deg)[::-1][:top_k]
    lines  = [f"  [ Top-{top_k} highest-degree nodes ]"]
    lines.append(f"    {'node_id':>8}  {'degree':>8}  {'rho_bar':>10}  {'dd(i)':>8}")
    lines.append("    " + "-" * 42)
    for i in idx:
        lines.append(f"    {i:>8,}  {deg[i]:>8,}  {rho[i]:>10.4f}  {dd[i]:>8.4f}")
    return lines


def main():
    args = parse_args()
    ckpt = torch.load(args.inp, map_location="cpu", weights_only=False)

    DD          = float(ckpt["DD_proxy"])
    dd_t        = ckpt["dd_per_node"]
    rho_t       = ckpt["rho_bar"]
    deg_t       = ckpt["deg"]
    dataset     = ckpt["dataset"]
    sample_nodes = ckpt["sample_nodes"]
    seed        = ckpt["seed"]
    min_deg     = ckpt["min_deg"]

    valid  = ~dd_t.isnan()
    dd     = dd_t[valid].numpy()
    rho    = rho_t[valid].numpy()
    deg    = deg_t[valid].numpy()

    n_total = len(dd_t)
    n_valid = int(valid.sum())

    low_frac  = float((dd < 0.33).mean())
    mod_frac  = float(((dd >= 0.33) & (dd < 0.66)).mean())
    high_frac = float((dd >= 0.66).mean())

    if DD < 0.33:
        demand_label = "low  (<0.33)"
    elif DD < 0.66:
        demand_label = "moderate  (0.33–0.66)"
    else:
        demand_label = "high  (>0.66)"

    sep = "=" * 60
    lines = [
        sep,
        "  DD_proxy SUMMARY",
        sep,
        "",
        f"  Dataset       : {dataset}",
        f"  Sample nodes  : {sample_nodes:,}  (seed={seed})",
        f"  min_deg       : {min_deg}",
        f"  Nodes total   : {n_total:,}",
        f"  Nodes valid   : {n_valid:,}  ({n_valid/n_total:.1%} of sample)",
        "",
        f"  DD_proxy(G)   : {DD:.6f}",
        f"  Demand level  : {demand_label}",
        "",
        "  [ Demand level breakdown — fraction of valid nodes ]",
        f"    {'low    (dd < 0.33)':<28}: {low_frac:.4f}  ({int(low_frac*n_valid):,} nodes)",
        f"    {'moderate (0.33 ≤ dd < 0.66)':<28}: {mod_frac:.4f}  ({int(mod_frac*n_valid):,} nodes)",
        f"    {'high   (dd ≥ 0.66)':<28}: {high_frac:.4f}  ({int(high_frac*n_valid):,} nodes)",
        "",
    ]

    lines += dist_stats(dd,  "dd(i) distribution") + [""]
    lines += dist_stats(rho, "rho_bar distribution") + [""]
    lines += degree_band_stats(dd, deg) + [""]
    lines += hub_spotlight(dd, rho, deg, args.top_hubs) + [""]
    lines.append(sep)

    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary  → {out_txt}")

    out_csv = Path(args.out_csv)
    df = pd.DataFrame({
        "node_id": np.where(valid.numpy())[0],
        "degree":  deg.astype(int),
        "rho_bar": rho,
        "dd":      dd,
    })
    df.to_csv(out_csv, index=False)
    print(f"Per-node → {out_csv}")


if __name__ == "__main__":
    main()
