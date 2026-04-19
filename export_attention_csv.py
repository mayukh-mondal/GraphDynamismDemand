import functools
from pathlib import Path

import torch
import numpy as np
import pandas as pd

_orig = torch.load
@functools.wraps(_orig)
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig(*args, **kwargs)
torch.load = _patched_load

PT_DIR  = Path("./results")
OUT_DIR = Path("./results/attention_tables")
LAYERS  = [0, 1]


def export_layer(layer: int):
    d = torch.load(PT_DIR / f"attention_comparison_layer{layer}.pt")

    alpha_gat   = d["alpha_gat"].numpy()    # [E, H]
    alpha_gatv2 = d["alpha_gatv2"].numpy()  # [E, H]
    edge_cos    = d["edge_cos"].numpy()      # [E]
    node_cos    = d["node_cos"].numpy()      # [N]
    node_jsd    = d["node_jsd"].numpy()      # [N]
    node_spear  = d["node_spearman"].numpy() # [N]

    E, H = alpha_gat.shape
    edge_dict = {"edge_index": np.arange(E)}
    for h in range(H):
        edge_dict[f"alpha_gat_h{h}"]   = alpha_gat[:, h]
        edge_dict[f"alpha_gatv2_h{h}"] = alpha_gatv2[:, h]
    edge_dict["edge_cosine"] = edge_cos
    pd.DataFrame(edge_dict).to_csv(OUT_DIR / f"edges_layer{layer}.csv", index=False)

    N = node_cos.shape[0]
    pd.DataFrame({
        "node":          np.arange(N),
        "deg":           d["deg"].numpy(),
        "node_cos":      node_cos,
        "node_jsd":      node_jsd,
        "node_spearman": node_spear,
    }).to_csv(OUT_DIR / f"nodes_layer{layer}.csv", index=False)


OUT_DIR.mkdir(parents=True, exist_ok=True)
for layer in LAYERS:
    export_layer(layer)
